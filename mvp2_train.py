# mvp_train_v2.py
from __future__ import annotations
import os, json, joblib, warnings
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from functools import lru_cache
from scipy.signal import butter, filtfilt, hilbert, welch, decimate as sp_decimate, savgol_filter
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ------------------- пути -------------------
BASE = "/Users/dmitrijnukin/PycharmProjects/PythonProject1/AImpulse"
RAW_DIR = f"{BASE}/data/raw"
LABELS_CSV = f"{BASE}/labels/labels_augmented.csv"
MODEL_DIR = f"{BASE}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- импорт пользовательских утилит (если есть) ---
try:
    from features import read_csv_3phase as _read3, extract_features_window as _extract
except Exception:
    _read3, _extract = None, None
    warnings.warn("features.py не найден. Использую фоллбэки.")

FS_RAW_DEFAULT = 25600.0

LABELS_ALL = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
TYPE2IDX = {c:i for i,c in enumerate(LABELS_ALL)}
IDX2TYPE = {i:c for c,i in TYPE2IDX.items()}
DEFECT_LABELS = ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
DEF2IDX = {c:i for i,c in enumerate(DEFECT_LABELS)}

# ---------------- доменные функции (PSD огибающей, энергии) ----------------
def bearing_lines(rpm, Z, d_mm, D_mm, theta_deg):
    fr = rpm / 60.0
    th = np.deg2rad(theta_deg)
    r = (d_mm / D_mm) * np.cos(th)
    FTF  = 0.5 * fr * (1 - r)
    BPFO = 0.5 * Z * fr * (1 - r)
    BPFI = 0.5 * Z * fr * (1 + r)
    BSF  = (D_mm / d_mm) * fr * 0.5 * (1 - r**2)
    return dict(fr=fr, FTF=FTF, BPFO=BPFO, BPFI=BPFI, BSF=BSF)

def envelope_psd(y, fs, fmax=320.0, mains_hz=50.0, bw=10.0):
    y = y - np.mean(y)
    lo = max(1.0, mains_hz - bw) / (fs/2)
    hi = min(fs/2 - 1.0, mains_hz + bw) / (fs/2)
    b, a = butter(4, [lo, hi], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, y)))
    f, P = welch(env, fs=fs, nperseg=min(2048, len(env)))
    m = f <= fmax
    return f[m], P[m]

def _area(f, P, m=None):
    if m is None: return float(np.trapezoid(P, f))
    return float(np.trapezoid(P[m], f[m]))

def band_energy(f, P, fc, bw=1.0, harmonics=1):
    if fc <= 0: return 0.0
    e = 0.0
    for k in range(1, harmonics+1):
        fck = fc * k
        bwk = max(bw, 0.02 * fck)
        m = (f >= fck - bwk) & (f <= fck + bwk)
        if m.any(): e += _area(f, P, m)
    return e

def sideband_energy(f, P, fc, fr, k=2, bw=1.0):
    """Сумма энергий в (fc ± m*fr), m=1..k."""
    if fc <= 0 or fr <= 0: return 0.0
    e = 0.0
    for m in range(1, k+1):
        for sign in (-1, +1):
            fcm = fc + sign * m * fr
            if fcm > 0:
                e += band_energy(f, P, fcm, bw=bw, harmonics=1)
    return e

def severity_from_bands(f,P,lines):
    numer = (band_energy(f,P,lines["BPFO"]) +
             band_energy(f,P,lines["BPFI"]) +
             band_energy(f,P,lines["BSF"])  +
             band_energy(f,P,lines["FTF"]))
    denom = _area(f, P) + 1e-12
    return float(np.clip(400.0 * numer/denom, 0, 100))

def defect_strength_ratio(f,P,lines):
    total = _area(f,P) + 1e-12
    val = (band_energy(f,P,lines["BPFO"]) +
           band_energy(f,P,lines["BPFI"]) +
           band_energy(f,P,lines["BSF"])  +
           band_energy(f,P,lines["FTF"])) / total
    return float(val)

def choose_phase_for_env(w, fs, lines, mains_hz=50.0, bw=10.0, fmax=320.0):
    best_idx, best_val = 0, -1.0
    for idx in range(min(w.shape[1], 3)):
        f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mains_hz, bw=bw)
        val = (band_energy(f,P,lines["BPFO"]) +
               band_energy(f,P,lines["BPFI"]) +
               band_energy(f,P,lines["BSF"])  +
               band_energy(f,P,lines["FTF"]))
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx

# ---------------- чтение/фичи ----------------
def read_csv_3phase(path: str) -> np.ndarray:
    if _read3 is not None:
        return _read3(path)
    df = pd.read_csv(path)
    arr = df.iloc[:, :3].to_numpy(dtype=np.float32)  # A,B,C
    return arr

def extract_features_window(w: np.ndarray, fs: float) -> np.ndarray:
    if _extract is not None:
        return _extract(w, fs)
    # базовые статистики по фазам + корреляции
    def stats(x):
        rms = np.sqrt(np.mean(x**2))
        mu = np.mean(x)
        sig = np.std(x) + 1e-9
        skew = np.mean(((x-mu)/sig)**3)
        kurt = np.mean(((x-mu)/sig)**4)
        crest = np.max(np.abs(x)) / (rms + 1e-9)
        return [rms, skew, kurt, crest]
    feats = []
    for k in range(min(3, w.shape[1])):
        feats += stats(w[:,k])
    if w.shape[1] >= 2:
        feats.append(np.corrcoef(w[:,0], w[:,1])[0,1])
    if w.shape[1] == 3:
        feats.append(np.corrcoef(w[:,0], w[:,2])[0,1])
        feats.append(np.corrcoef(w[:,1], w[:,2])[0,1])
    return np.array(feats, dtype=np.float32)

@dataclass
class FeatureCfg:
    fs_raw: float = FS_RAW_DEFAULT
    mains_hz: float = 50.0
    mains_bw: float = 10.0
    fmax_env: float = 320.0

def decimate_to_fs(x, fs_raw, fs_target):
    factor = int(round(fs_raw / fs_target))
    if factor <= 1:
        return x.astype(np.float32), float(fs_raw)
    y = np.zeros((int(np.ceil(x.shape[0]/factor)), x.shape[1]), dtype=np.float32)
    for i in range(x.shape[1]):
        y[:, i] = sp_decimate(x[:, i], factor, ftype='iir', zero_phase=True)
    return y.astype(np.float32), float(fs_raw/factor)

@lru_cache(maxsize=None)
def _load_decimated_cached(file_basename: str, fs_target: float):
    path = os.path.join(RAW_DIR, file_basename)
    x_raw = read_csv_3phase(path)
    x, fs = decimate_to_fs(x_raw, FS_RAW_DEFAULT, fs_target)
    return x, fs

def make_features_for_row(row: dict, cfg: FeatureCfg) -> Tuple[np.ndarray, Dict]:
    fs_target = float(row["fs"])
    x, fs = _load_decimated_cached(row["file"], fs_target)
    i0, i1 = int(row["i0"]), int(row["i1"])
    w = x[i0:i1, :]

    f_base = extract_features_window(w, fs)

    lines = bearing_lines(row["rpm"], row["Z"], row["dmm"], row["Dmm"], row["theta"])
    ph = choose_phase_for_env(w, fs, lines, cfg.mains_hz, cfg.mains_bw, cfg.fmax_env)
    f_env, P_env = envelope_psd(w[:, ph], fs, fmax=cfg.fmax_env, mains_hz=cfg.mains_hz, bw=cfg.mains_bw)

    # относительные энергии (гребёнки) и боковые полосы
    def rel(fc, h=3):  # comb
        total = _area(f_env, P_env) + 1e-12
        return band_energy(f_env, P_env, fc, bw=1.0, harmonics=h) / total
    def sbr(fc):  # sideband ratio
        cen = band_energy(f_env, P_env, fc, bw=1.0, harmonics=1) + 1e-12
        sb1 = sideband_energy(f_env, P_env, fc, lines["fr"], k=1, bw=1.0)
        sb2 = sideband_energy(f_env, P_env, fc, lines["fr"], k=2, bw=1.0)
        return np.array([sb1/cen, sb2/cen], dtype=np.float32)

    f_dom = [rel(lines["FTF"],3), rel(lines["BPFO"],3), rel(lines["BPFI"],3), rel(lines["BSF"],3),
             rel(lines["fr"],2),  rel(2*lines["fr"],2),
             severity_from_bands(f_env, P_env, lines),
             defect_strength_ratio(f_env, P_env, lines)]
    # боковые полосы для подшипниковых:
    for key in ["FTF","BPFO","BPFI","BSF"]:
        f_dom += list(sbr(lines[key]))

    f_dom = np.array(f_dom, dtype=np.float32)

    feats = np.concatenate([f_base, f_dom], axis=0)
    meta = {"chosen_phase": ph}
    return feats, meta

# ---------------- сбор набора фич ----------------
def build_dataset(labels_csv: str, cfg: FeatureCfg):
    df = pd.read_csv(labels_csv)
    df = df[df["y_type"].isin(LABELS_ALL)].copy()
    df["y_bin"] = (df["y_type"]!="normal").astype(int)
    df["y_multi"] = df["y_type"].map(TYPE2IDX)

    # безопасное приведение типов
    cast_int = ["Z","i0","i1","theta"]
    for c in ["rpm","dmm","Dmm","fs","t0","t1"]: df[c] = df[c].astype(float)
    for c in cast_int: df[c] = df[c].astype(int)

    X, metas = [], []
    for i, row in enumerate(df.to_dict(orient="records"), 1):
        feats, meta = make_features_for_row(row, cfg)
        X.append(feats); metas.append(meta)
        if i % 200 == 0 or i == len(df):
            print(f"... извлечено {i}/{len(df)} окон")
    X = np.vstack(X).astype(np.float32)
    y_bin = df["y_bin"].to_numpy(int)
    y_multi = df["y_multi"].to_numpy(int)
    y_sev = df["severity"].to_numpy(float)
    groups = df["file"].astype(str).to_numpy()
    return X, y_bin, y_multi, y_sev, groups, df, metas

# ---------------- модель-обёртка ----------------
@dataclass
class MVPModel:
    clf_bin: HistGradientBoostingClassifier
    calib_bin: object
    clf_def: HistGradientBoostingClassifier
    reg_sev: HistGradientBoostingRegressor
    scaler: StandardScaler
    label_map: Dict[int,str]
    thr_bin: float = 0.5
    tau_defclass: float = 0.30   # порог уверенности дефект-класса

    def predict_all(self, X: np.ndarray):
        Xs = self.scaler.transform(X)

        # детектор → калиброванные вероятности дефекта
        p_raw = np.clip(self.clf_bin.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
        p_def = self.calib_bin.predict(p_raw)

        y_bin_hat = (p_def >= self.thr_bin).astype(int)

        # по умолчанию — normal
        p_multi7 = np.zeros((Xs.shape[0], 7), dtype=float)
        y_multi_hat = np.full(Xs.shape[0], TYPE2IDX["normal"], dtype=int)
        sev_hat = np.zeros(Xs.shape[0], dtype=float)

        # обрабатываем только те, где детектор сказал "дефект"
        idx = np.where(y_bin_hat == 1)[0]
        if len(idx) > 0:
            p_def6 = self.clf_def.predict_proba(Xs[idx])  # [n_def, 6]
            # переносим в 7-классовое пространство
            for j, name in enumerate(DEFECT_LABELS):
                p_multi7[idx, TYPE2IDX[name]] = p_def6[:, j]

            # нормализуем дефектные вероятности
            s = p_multi7[idx].sum(axis=1, keepdims=True) + 1e-12
            p_multi7[idx] = p_multi7[idx] / s

            # "reject option": если уверенность низкая — считаем normal
            conf = p_multi7[idx].max(axis=1)
            keep = conf >= self.tau_defclass
            idx_keep = idx[keep]
            idx_reject = idx[~keep]

            if len(idx_keep) > 0:
                y_multi_hat[idx_keep] = np.argmax(p_multi7[idx_keep], axis=1)
                sev_hat[idx_keep] = np.clip(self.reg_sev.predict(Xs[idx_keep]), 0, 100)

            # отвергнутым явно ставим normal
            if len(idx_reject) > 0:
                p_multi7[idx_reject, TYPE2IDX["normal"]] = 1.0

        # тем, кто сразу прошёл как normal
        p_multi7[y_bin_hat == 0, TYPE2IDX["normal"]] = 1.0

        return y_bin_hat, p_def, y_multi_hat, p_multi7, sev_hat

def pick_threshold(y_true_bin, p_scores, target_fpr=None, grid=None):
    """
    Подбираем порог для детектора, максимизируя Balanced Accuracy
    (BA = (TPR + TNR)/2). Это жёстче наказывает и за FP, и за FN.
    Если target_fpr задан, фильтруем пороги с FPR <= target_fpr.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)  # шаг 0.01

    best_t, best_ba = None, -1.0
    cand = []

    for t in grid:
        y_hat = (p_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_hat, labels=[0,1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        fpr = 1.0 - tnr
        ba  = 0.5 * (tpr + tnr)
        cand.append((t, ba, fpr))

    if target_fpr is not None:
        cand = [c for c in cand if c[2] <= target_fpr]

    if not cand:
        cand = [(t, ba, fpr) for t, ba, fpr in cand]  # на случай пустого фильтра — не трогаем

    for t, ba, _ in cand:
        if ba > best_ba:
            best_ba, best_t = ba, float(t)

    # запасной план: если вдруг best_t так и не определился
    if best_t is None:
        best_t = float(grid[int(np.argmax([ba for _, ba, _ in cand]))])

    return best_t

def pick_threshold_recall(y_true_bin, p_scores, target_recall=0.80):
    """
    Подбираем порог так, чтобы полнота по дефектам (TPR/recall) была >= target_recall.
    Среди подходящих порогов выбираем с минимальным FPR.
    Если достичь не удаётся — fallback на максимальную Balanced Accuracy.
    """
    grid = np.linspace(0.01, 0.99, 99)

    # 1) найти порог с требуемой полнотой и минимальным FPR
    best_t, best_fpr = None, 1.0
    for t in grid:
        y_hat = (p_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_hat, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)      # recall дефектов
        fpr = fp / (fp + tn + 1e-12)
        if tpr >= target_recall and fpr < best_fpr:
            best_fpr, best_t = fpr, float(t)

    if best_t is not None:
        return best_t

    # 2) fallback — максимизация balanced accuracy
    best_t, best_ba = None, -1.0
    for t in grid:
        y_hat = (p_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_hat, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)

    return best_t

def file_majority(y_true, y_pred, files):
    out = {}
    for f in np.unique(files):
        m = (files == f)
        # большинство по y_pred
        vals, cnts = np.unique(y_pred[m], return_counts=True)
        yhat_f = int(vals[np.argmax(cnts)])
        # истинный – тоже большинством
        vals_t, cnts_t = np.unique(y_true[m], return_counts=True)
        ytrue_f = int(vals_t[np.argmax(cnts_t)])
        out[f] = (ytrue_f, yhat_f)
    yt = np.array([v[0] for v in out.values()])
    yp = np.array([v[1] for v in out.values()])
    return yt, yp

def file_prob_aggregate(y_true, p_multi, files):
    """
    По каждому файлу усредняем вероятности p_multi по окнам
    и берём argmax. Истинный класс файла — большинством y_true.
    """
    out = {}
    for f in np.unique(files):
        m = (files == f)
        ytrue_f = int(np.bincount(y_true[m]).argmax())
        p_mean = p_multi[m].mean(axis=0)       # усредняем вероятности по окнам
        yhat_f = int(np.argmax(p_mean))        # предсказываем класс файла
        out[f] = (ytrue_f, yhat_f)
    yt = np.array([v[0] for v in out.values()])
    yp = np.array([v[1] for v in out.values()])
    return yt, yp

COARSE_MAP = {
    TYPE2IDX["normal"]:        0,     # normal
    TYPE2IDX["BPFO"]:          1,     # bearing faults
    TYPE2IDX["BPFI"]:          1,
    TYPE2IDX["BSF"]:           1,
    TYPE2IDX["FTF"]:           1,
    TYPE2IDX["imbalance"]:     2,     # alignment / imbalance
    TYPE2IDX["misalignment"]:  2,
}
COARSE_NAMES = ["normal", "bearing", "align/imbalance"]

def to_coarse(y7: np.ndarray) -> np.ndarray:
    return np.array([COARSE_MAP[int(v)] for v in y7], dtype=int)

def file_prob_aggregate_coarse(y_true7: np.ndarray, p_multi7: np.ndarray, files: np.ndarray):
    # сворачиваем вероятности 7→3 и агрегируем по файлу усреднением вероятностей
    P = np.zeros((p_multi7.shape[0], 3), dtype=float)
    P[:, 0] = p_multi7[:, TYPE2IDX["normal"]]
    P[:, 1] = (p_multi7[:, TYPE2IDX["BPFO"]] + p_multi7[:, TYPE2IDX["BPFI"]] +
               p_multi7[:, TYPE2IDX["BSF"]]  + p_multi7[:, TYPE2IDX["FTF"]])
    P[:, 2] = (p_multi7[:, TYPE2IDX["imbalance"]] + p_multi7[:, TYPE2IDX["misalignment"]])
    yt3 = to_coarse(y_true7)
    return file_prob_aggregate(yt3, P, files)

# ---------------- тренировочное ядро (CV + подбор порога) ----------------
def train_and_eval(X, y_bin, y_multi, y_sev, groups, n_splits=5, random_state=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    N = len(y_bin)

    # 7 -> 6 (без 'normal')
    idx_to_def = {TYPE2IDX[name]: DEF2IDX[name] for name in DEFECT_LABELS}
    y_def_full = np.full(N, -1, dtype=int)
    m_def = (y_bin == 1)
    def_idx = np.where(m_def)[0]
    for i in def_idx:
        y_def_full[i] = idx_to_def[y_multi[i]]

    # ----- веса бинарки (строго balanced, без доп. усиления normal) -----
    classes_bin = np.unique(y_bin)
    cb = compute_class_weight(class_weight='balanced', classes=classes_bin, y=y_bin)
    w_bin_map = {c: w for c, w in zip(classes_bin, cb)}
    w_bin_full = np.array([w_bin_map[y] for y in y_bin], dtype=float)
    w_bin_full[y_bin == 0] *= 1.5

    # ----- веса 6-классов (только дефекты) -----
    classes_def = np.unique(y_def_full[y_def_full >= 0])
    cw_def = compute_class_weight(class_weight='balanced',
                                  classes=classes_def,
                                  y=y_def_full[y_def_full >= 0])
    w_def_map = {c: w for c, w in zip(classes_def, cw_def)}
    w_def_full = np.zeros(N, dtype=float)
    for c, w in w_def_map.items():
        w_def_full[y_def_full == c] = w

    gkf = GroupKFold(n_splits=n_splits)
    f1_bin, f1_macro, mae_sev, rmse_sev = [], [], [], []
    cms, thr_list = [], []

    for fold, (tr, te) in enumerate(gkf.split(Xs, y_multi, groups)):
        # ----- детектор -----
        clf_bin = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=300,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            random_state=random_state
        )
        clf_bin.fit(Xs[tr], y_bin[tr], sample_weight=w_bin_full[tr])

        # Platt-калибровка: логистика по сырым p_train
        p_tr_raw = np.clip(clf_bin.predict_proba(Xs[tr])[:, 1], 1e-6, 1 - 1e-6)
        calib = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        calib.fit(p_tr_raw, y_bin[tr], sample_weight=w_bin_full[tr])

        p_te_raw = np.clip(clf_bin.predict_proba(Xs[te])[:, 1], 1e-6, 1 - 1e-6)
        p_te = calib.predict(p_te_raw)
        thr_star = pick_threshold_recall(y_bin[te], p_te, target_recall=0.72)
        thr_list.append(thr_star)
        yb_hat = (p_te >= thr_star).astype(int)

        # ----- 6-классовый классификатор (только дефекты) -----
        idx_def_tr = tr[y_bin[tr] == 1]
        clf_def = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=500,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            random_state=random_state
        ).fit(Xs[idx_def_tr], y_def_full[idx_def_tr], sample_weight=w_def_full[idx_def_tr])

        # ----- регрессор severity (только дефекты) -----
        reg_sev = HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=600, l2_regularization=1e-3,
            early_stopping=True, n_iter_no_change=30, validation_fraction=0.1,
            random_state=random_state
        ).fit(Xs[idx_def_tr], y_sev[idx_def_tr])

        # ----- инференс на te -----
        y_final = np.full(te.shape[0], TYPE2IDX["normal"], dtype=int)
        p_multi7 = np.zeros((te.shape[0], 7), dtype=float)

        idx_def_te = np.where(yb_hat == 1)[0]
        if len(idx_def_te) > 0:
            p6 = clf_def.predict_proba(Xs[te][idx_def_te])
            for j, name in enumerate(DEFECT_LABELS):
                p_multi7[idx_def_te, TYPE2IDX[name]] = p6[:, j]
            s = p_multi7[idx_def_te].sum(axis=1, keepdims=True) + 1e-12
            p_multi7[idx_def_te] /= s
            y_final[idx_def_te] = np.argmax(p_multi7[idx_def_te], axis=1)

        ys_hat = np.zeros(te.shape[0], dtype=float)
        if len(idx_def_te) > 0:
            ys_hat[idx_def_te] = np.clip(reg_sev.predict(Xs[te][idx_def_te]), 0, 100)

        f1b = f1_score(y_bin[te], yb_hat)
        f1m = f1_score(y_multi[te], y_final, average='macro')
        mae = mean_absolute_error(y_sev[te], ys_hat)
        rmse = np.sqrt(mean_squared_error(y_sev[te], ys_hat))
        cm = confusion_matrix(y_multi[te], y_final, labels=np.arange(len(LABELS_ALL)))

        f1_bin.append(f1b); f1_macro.append(f1m); mae_sev.append(mae); rmse_sev.append(rmse); cms.append(cm)
        print(f"[Fold {fold+1}] thr={thr_star:.3f} | F1-bin={f1b:.3f} | F1-macro7={f1m:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

    print("\n=== CV MEAN ± STD ===")
    print(f"F1 (defect)         : {np.mean(f1_bin):.3f} ± {np.std(f1_bin):.03f}")
    print(f"F1 (macro, 7-class) : {np.mean(f1_macro):.3f} ± {np.std(f1_macro):.03f}")
    print(f"MAE(severity)       : {np.mean(mae_sev):.2f} ± {np.std(mae_sev):.02f}")
    print(f"RMSE(severity)      : {np.mean(rmse_sev):.2f} ± {np.std(rmse_sev):.02f}")

    # ----- финальные модели на всём train + итоговый порог -----
    thr_final = float(np.median(thr_list))

    clf_bin_f = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=300,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
        random_state=random_state
    ).fit(Xs, y_bin, sample_weight=w_bin_full)

    # финальный калибратор на всём train
    p_all_raw = np.clip(clf_bin_f.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    calib_f = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calib_f.fit(p_all_raw, y_bin, sample_weight=w_bin_full)

    clf_def_f = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=500,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
        random_state=random_state
    ).fit(Xs[def_idx], y_def_full[def_idx], sample_weight=w_def_full[def_idx])

    reg_sev_f = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.06, max_iter=600, l2_regularization=1e-3,
        early_stopping=True, n_iter_no_change=30, validation_fraction=0.1,
        random_state=random_state
    ).fit(Xs[def_idx], y_sev[def_idx])

    model = MVPModel(clf_bin_f, calib_f, clf_def_f, reg_sev_f, scaler, IDX2TYPE, thr_bin=thr_final)
    print("[DEBUG] Calib probs (train) min/median/max:",
          float(p_all_raw.min()), float(np.median(p_all_raw)), float(p_all_raw.max()))
    p_all_cal = calib_f.predict(p_all_raw)  # IsotonicRegression -> 1D вероятности
    print("[DEBUG] After calib min/median/max:",
          float(p_all_cal.min()), float(np.median(p_all_cal)), float(p_all_cal.max()))
    return model, cms, thr_final

# ---------------- прогноз TTF для одного испытания ----------------
def forecast_ttf_for_file(sev_times: np.ndarray, t_seconds: np.ndarray, thr: float = 80.0) -> float:
    if len(sev_times) < 3:
        return np.inf
    s = savgol_filter(sev_times, window_length=min(11, len(sev_times)//2*2+1), polyorder=2) \
        if len(sev_times) >= 11 else sev_times
    k = max(5, int(0.3*len(s)))
    y = s[-k:]; x = t_seconds[-k:]
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if a <= 1e-6:
        return np.inf
    t_hit = (thr - b) / a
    now = t_seconds[-1]
    return float(max(0.0, t_hit - now)) if t_hit > now else 0.0

# ---------------- main ----------------
def main():
    cfg = FeatureCfg(fs_raw=FS_RAW_DEFAULT, mains_hz=50.0, mains_bw=10.0, fmax_env=320.0)
    print("[1/5] Сбор фич...")
    X, y_bin, y_multi, y_sev, groups, df_lab, metas = build_dataset(LABELS_CSV, cfg)

    print(f"[INFO] Samples: {len(df_lab)}; Features: {X.shape[1]}; Files: {df_lab['file'].nunique()}")
    print(df_lab['y_type'].value_counts())

    # ---------- HOLD-OUT: 20% файлов на тест ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_multi, groups))
    train_files = sorted(set(groups[train_idx]))
    test_files  = sorted(set(groups[test_idx]))
    print(f"[INFO] Train files: {len(train_files)} | Test files: {len(test_files)}")
    print("       Test hold-out:", ", ".join(map(str, test_files)))

    # ---------- [2/5] Обучение + CV (только train) ----------
    print("[2/5] Обучение + CV...")
    model, cms, thr = train_and_eval(
        X[train_idx], y_bin[train_idx], y_multi[train_idx], y_sev[train_idx], groups[train_idx],
        n_splits=5, random_state=42
    )
    print(f"[INFO] Итоговый порог детектора (median OOF): {thr:.3f}")

    # ---------- [3/5] Оценка на hold-out ----------
    print("[3/5] Оценка на hold-out...")
    yb_hat, p_bin, ym_hat, p_multi, ys_hat = model.predict_all(X[test_idx])
    f1b  = f1_score(y_bin[test_idx], yb_hat)
    f1m  = f1_score(y_multi[test_idx], ym_hat, average='macro')
    mae  = mean_absolute_error(y_sev[test_idx], ys_hat)
    rmse = np.sqrt(mean_squared_error(y_sev[test_idx], ys_hat))
    cm   = confusion_matrix(y_multi[test_idx], ym_hat, labels=np.arange(len(LABELS_ALL)))
    print("\n=== HOLD-OUT TEST ===")
    print(f"F1 (defect)         : {f1b:.3f}")
    print(f"F1 (macro, 7-class) : {f1m:.3f}")
    print(f"MAE(severity)       : {mae:.2f}")
    print(f"RMSE(severity)      : {rmse:.2f}")
    print("Confusion matrix order:", LABELS_ALL)
    print(cm)
    print(classification_report(y_multi[test_idx], ym_hat, target_names=LABELS_ALL, digits=3, zero_division=0))

    # --- агрегируем по файлам: большинство по окнам ---
    files_holdout = groups[test_idx].astype(str)

    # 7 классов: вероятностная агрегация
    yt_f, yp_f = file_prob_aggregate(y_multi[test_idx], p_multi, files_holdout)
    print("[FILES] macro-F1 (7 классов, prob-avg по файлам):",
          f1_score(yt_f, yp_f, average="macro"))

    ytb_f, ypb_f = file_majority(y_bin[test_idx], yb_hat, files_holdout)
    print("[FILES] F1 (defect vs normal, по файлам):", f1_score(ytb_f, ypb_f))

    # ---- ДОБАВЬ вот это: 3 укрупнённых класса по файлам ----
    yt3_f, yp3_f = file_prob_aggregate_coarse(y_multi[test_idx], p_multi, files_holdout)
    print("[FILES-3cls] macro-F1 (normal/bearing/align):",
          f1_score(yt3_f, yp3_f, average="macro"))

    # сохраним предсказания hold-out
    holdout = df_lab.iloc[test_idx][["file","t0","t1","y_type"]].copy()
    holdout["p_defect"] = p_bin
    holdout["y_pred"] = [IDX2TYPE[i] for i in ym_hat]
    holdout["sev_hat"] = ys_hat
    holdout.to_csv(f"{MODEL_DIR}/holdout_predictions_v2.csv", index=False)
    print(f"[INFO] Hold-out предсказания: {MODEL_DIR}/holdout_predictions_v2.csv")

    # ---------- [4/5] Сохранение модели ----------
    print("[4/5] Сохранение модели...")
    joblib.dump(model, f"{MODEL_DIR}/mvp_model_v2.joblib")
    with open(f"{MODEL_DIR}/features_info_v2.json","w") as f:
        json.dump({
            "features_dim": int(X.shape[1]),
            "labels": LABELS_ALL,
            "defect_labels": DEFECT_LABELS,
            "train_files": train_files, "test_files": test_files,
            "bin_threshold": float(thr)
        }, f, ensure_ascii=False, indent=2)

    # ---------- [5/5] Демонстрация TTF на одном тестовом файле ----------
    if len(test_files) > 0:
        file0 = test_files[0]
        m = (df_lab['file'].astype(str).values == file0)
        idx = np.argsort(df_lab.loc[m, 't0'].values)
        sev_hat_demo = model.predict_all(X[m])[4][idx]
        t_seq = df_lab.loc[m, 't0'].values[idx]
        ttf_sec = forecast_ttf_for_file(np.clip(sev_hat_demo,0,100), t_seq, thr=80.0)
        print(f"[DEMO] TTF до 80 для файла {file0}: {ttf_sec:.1f} сек")

if __name__ == "__main__":
    main()
