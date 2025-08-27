# mvp_train.py
from __future__ import annotations
import os, json, joblib, warnings
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.signal import butter, filtfilt, hilbert, welch, decimate as sp_decimate, savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from functools import lru_cache
from sklearn.model_selection import GroupShuffleSplit



# --- ПУТИ (под тебя) ---
BASE = "/Users/dmitrijnukin/PycharmProjects/PythonProject1/AImpulse"
RAW_DIR = f"{BASE}/data/raw"
LABELS_CSV = f"{BASE}/labels/labels_augmented.csv"   # <- твой свежий
MODEL_DIR = f"{BASE}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- ИМПОРТ твоих утилит (если нет — используем фоллбэки ниже) ---
try:
    from features import read_csv_3phase as _read3, extract_features_window as _extract
except Exception:
    _read3, _extract = None, None
    warnings.warn("features.py не найден. Будут использованы простые фоллбэки.")

FS_RAW_DEFAULT = 25600.0   # как в разметчике

LABELS = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
TYPE2IDX = {c:i for i,c in enumerate(LABELS)}
IDX2TYPE = {i:c for c,i in TYPE2IDX.items()}

# ---------------- доменные функции (огибающая, линии, энергии) ----------------
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

def band_energy(f, P, fc, bw=2.0, harmonics=1):
    if fc <= 0: return 0.0
    e = 0.0
    for k in range(1, harmonics+1):
        fck = fc * k
        bwk = max(bw, 0.02 * fck)
        m = (f >= fck - bwk) & (f <= fck + bwk)
        if m.any(): e += _area(f, P, m)
    return e

def severity_from_bands(f,P,lines):
    bands = ["BPFO","BPFI","BSF","FTF"]
    numer = sum(band_energy(f,P,lines[b]) for b in bands)
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

# ---------------- фоллбэки чтения и базовых фич ----------------
def read_csv_3phase(path: str) -> np.ndarray:
    if _read3 is not None:
        return _read3(path)
    df = pd.read_csv(path)
    arr = df.iloc[:, :3].to_numpy(dtype=np.float32)  # A,B,C первые 3 колонки
    return arr

def extract_features_window(w: np.ndarray, fs: float) -> np.ndarray:
    if _extract is not None:
        return _extract(w, fs)
    # простой фоллбэк: RMS/Skew/Kurt/Crest-factor по фазам + межфазные корреляции
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
    # межфазные корреляции
    if w.shape[1] >= 2:
        feats.append(np.corrcoef(w[:,0], w[:,1])[0,1])
    if w.shape[1] == 3:
        feats.append(np.corrcoef(w[:,0], w[:,2])[0,1])
        feats.append(np.corrcoef(w[:,1], w[:,2])[0,1])
    return np.array(feats, dtype=np.float32)

# ---------------- извлечение фич из одной строки labels ----------------
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

def make_features_for_row(row, cfg: FeatureCfg) -> Tuple[np.ndarray, Dict]:
    # читаем CSV и приводим к fs из labels
    fs_target = float(row["fs"])
    x, fs = _load_decimated_cached(row["file"], fs_target)
    i0, i1 = int(row["i0"]), int(row["i1"])
    w = x[i0:i1, :]  # окно Nx3

    # базовые фичи из твоего features.py (или фоллбэк)
    f_base = extract_features_window(w, fs)

    # доменные фичи по лучшей фазе
    lines = bearing_lines(row["rpm"], row["Z"], row["dmm"], row["Dmm"], row["theta"])
    ph = choose_phase_for_env(w, fs, lines, cfg.mains_hz, cfg.mains_bw, cfg.fmax_env)
    f_env, P_env = envelope_psd(w[:, ph], fs, fmax=cfg.fmax_env, mains_hz=cfg.mains_hz, bw=cfg.mains_bw)
    def _rel(fc, h=3):
        total = _area(f_env, P_env) + 1e-12
        return band_energy(f_env, P_env, fc, bw=1.0, harmonics=h) / total
    f_dom = np.array([
        _rel(lines["FTF"], 3), _rel(lines["BPFO"], 3), _rel(lines["BPFI"], 3), _rel(lines["BSF"], 3),
        _rel(lines["fr"], 2),  _rel(2*lines["fr"], 2),
        severity_from_bands(f_env, P_env, lines),       # как в разметчике (0..100)
        defect_strength_ratio(f_env, P_env, lines),     # доля подшипн. полос
    ], dtype=np.float32)

    # итоговый вектор
    feats = np.concatenate([f_base, f_dom], axis=0)
    meta = {"chosen_phase": ph}
    return feats, meta

# ---------------- сбор набора фич ----------------
def build_dataset(labels_csv: str, cfg: FeatureCfg):
    df = pd.read_csv(labels_csv)
    # фильтруем корректные классы
    df = df[df["y_type"].isin(LABELS)].copy()
    df["y_bin"] = (df["y_type"]!="normal").astype(int)
    df["y_multi"] = df["y_type"].map(TYPE2IDX)
    # гарантируем типы
    for c in ["rpm","Z","dmm","Dmm","theta","fs","i0","i1","t0","t1"]:
        df[c] = df[c].astype(float if c not in ["Z","i0","i1","theta"] else int, errors='ignore')

    X, metas = [], []
    for i, r in enumerate(df.itertuples(index=False), 1):
        feats, meta = make_features_for_row(r._asdict(), cfg)
        X.append(feats);
        metas.append(meta)
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
    clf_multi: HistGradientBoostingClassifier
    reg_sev: HistGradientBoostingRegressor
    scaler: StandardScaler
    label_map: Dict[int,str]

    def predict_all(self, X: np.ndarray):
        Xs = self.scaler.transform(X)
        p_bin = self.clf_bin.predict_proba(Xs)[:,1]
        y_bin_hat = (p_bin >= 0.5).astype(int)
        p_multi = self.clf_multi.predict_proba(Xs)  # shape [N,7]
        y_multi_hat = np.argmax(p_multi, axis=1)
        sev_hat = np.clip(self.reg_sev.predict(Xs), 0, 100)
        return y_bin_hat, p_bin, y_multi_hat, p_multi, sev_hat

# ---------------- тренировка/оценка ----------------
def train_and_eval(X, y_bin, y_multi, y_sev, groups, n_splits=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Веса для мультикласса
    classes = np.unique(y_multi)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_multi)
    sample_w_multi = np.array([cw[np.where(classes==yy)[0][0]] for yy in y_multi])

    # Бинарке тоже дадим веса (устойчивый вариант)
    classes_bin = np.unique(y_bin)  # np.ndarray
    cb = compute_class_weight(class_weight='balanced', classes=classes_bin, y=y_bin)
    w_bin_map = {c: w for c, w in zip(classes_bin, cb)}
    sample_w_bin = np.array([w_bin_map[y] for y in y_bin], dtype=float)

    gkf = GroupKFold(n_splits=n_splits)
    f1_bin, f1_macro, mae_sev, rmse_sev = [], [], [], []
    cms = []

    for fold, (tr, te) in enumerate(gkf.split(Xs, y_multi, groups)):
        Xtr, Xte = Xs[tr], Xs[te]
        yb_tr, yb_te = y_bin[tr], y_bin[te]
        ym_tr, ym_te = y_multi[tr], y_multi[te]
        ys_tr, ys_te = y_sev[tr], y_sev[te]

        w_bin_tr = sample_w_bin[tr]
        w_multi_tr = sample_w_multi[tr]

        clf_bin = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=300,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            random_state=42
        )
        clf_multi = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=500,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            random_state=42
        )
        reg_sev = HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=600, l2_regularization=1e-3,
            early_stopping=True, n_iter_no_change=30, validation_fraction=0.1,
            random_state=42
        )

        clf_bin.fit(Xtr, yb_tr, sample_weight=w_bin_tr)
        clf_multi.fit(Xtr, ym_tr, sample_weight=w_multi_tr)
        reg_sev.fit(Xtr, ys_tr)

        p_bin = clf_bin.predict_proba(Xte)[:,1]
        yb_hat = (p_bin>=0.5).astype(int)
        ym_hat = clf_multi.predict(Xte)
        ys_hat = np.clip(reg_sev.predict(Xte), 0, 100)

        f1b = f1_score(yb_te, yb_hat)
        f1m = f1_score(ym_te, ym_hat, average='macro')
        mae = mean_absolute_error(ys_te, ys_hat)
        mse = mean_squared_error(ys_te, ys_hat)
        rmse = np.sqrt(mse)

        f1_bin.append(f1b); f1_macro.append(f1m); mae_sev.append(mae); rmse_sev.append(rmse)
        cms.append(confusion_matrix(ym_te, ym_hat, labels=np.arange(len(LABELS))))

        print(f"[Fold {fold+1}] F1-bin={f1b:.3f} | F1-macro7={f1m:.3f} | MAE_sev={mae:.2f} | RMSE_sev={rmse:.2f}")

    print("\n=== CV MEAN ± STD ===")
    print(f"F1 (defect)         : {np.mean(f1_bin):.3f} ± {np.std(f1_bin):.03f}")
    print(f"F1 (macro, 7-class) : {np.mean(f1_macro):.3f} ± {np.std(f1_macro):.03f}")
    print(f"MAE(severity)       : {np.mean(mae_sev):.2f} ± {np.std(mae_sev):.02f}")
    print(f"RMSE(severity)      : {np.mean(rmse_sev):.2f} ± {np.std(rmse_sev):.02f}")

    # обучим финальные на всём
    clf_bin_f = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=300,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.1
    )
    clf_multi_f = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=500,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.1
    )
    reg_sev_f = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.06, max_iter=600, l2_regularization=1e-3,
        early_stopping=True, n_iter_no_change=30, validation_fraction=0.1
    )
    clf_bin_f.fit(Xs, y_bin, sample_weight=sample_w_bin)
    clf_multi_f.fit(Xs, y_multi, sample_weight=sample_w_multi)
    reg_sev_f.fit(Xs, y_sev)

    model = MVPModel(clf_bin_f, clf_multi_f, reg_sev_f, scaler, IDX2TYPE)
    return model, cms

# ---------------- прогноз TTF по одному файлу ----------------
def forecast_ttf_for_file(sev_times: np.ndarray, t_seconds: np.ndarray, thr: float = 80.0) -> float:
    """
    Возвращает оценку времени (сек) до достижения порога thr.
    Используем сглаживание и линейную аппроксимацию последних точек.
    """
    if len(sev_times) < 3:
        return np.inf
    s = savgol_filter(sev_times, window_length=min(11, len(sev_times)//2*2+1), polyorder=2) \
        if len(sev_times) >= 11 else sev_times
    # берём последние 30% точек, но не меньше 5
    k = max(5, int(0.3*len(s)))
    y = s[-k:]; x = t_seconds[-k:]
    # линейная регрессия по МНК
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]  # sev ~ a*t + b
    if a <= 1e-6:
        return np.inf
    t_hit = (thr - b) / a
    now = t_seconds[-1]
    return float(max(0.0, t_hit - now)) if t_hit > now else 0.0

# ---------------- main ----------------
def main():
    cfg = FeatureCfg(fs_raw=FS_RAW_DEFAULT, mains_hz=50.0, mains_bw=10.0, fmax_env=320.0)
    print("[1/4] Сбор фич...")
    X, y_bin, y_multi, y_sev, groups, df_lab, metas = build_dataset(LABELS_CSV, cfg)

    print(f"[INFO] Samples: {len(df_lab)}; Features: {X.shape[1]}; Files: {df_lab['file'].nunique()}")
    print(df_lab['y_type'].value_counts())

    # ---------- HOLD-OUT: 20% файлов на тест ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_multi, groups))

    # (подстрахуемся: хотим, чтобы train покрывал все классы)
    seen_train = set(np.unique(y_multi[train_idx]))
    all_classes = set(np.unique(y_multi))
    if seen_train != all_classes:
        print("[WARN] В train отсутствуют классы:", [c for c in all_classes - seen_train],
              " — попробуй другой random_state или другой test_size.")
        # Можно перегенерировать сплит с другим seed, если захочешь.

    train_files = sorted(set(groups[train_idx]))
    test_files  = sorted(set(groups[test_idx]))
    print(f"[INFO] Train files: {len(train_files)} | Test files: {len(test_files)}")
    print("       Test hold-out (по файлам):", ", ".join(map(str, test_files)))

    # ---------- [2/4] Обучение + CV-только-на-TRAIN ----------
    print("[2/4] Обучение + CV (только train)...")
    model, cms = train_and_eval(
        X[train_idx], y_bin[train_idx], y_multi[train_idx], y_sev[train_idx], groups[train_idx],
        n_splits=5
    )

    # ---------- [3/4] Оценка на чистом TEST ----------
    print("[3/4] Оценка на hold-out тесте...")
    yb_hat, p_bin, ym_hat, p_multi, ys_hat = model.predict_all(X[test_idx])
    f1b  = f1_score(y_bin[test_idx], yb_hat)
    f1m  = f1_score(y_multi[test_idx], ym_hat, average='macro')
    mae  = mean_absolute_error(y_sev[test_idx], ys_hat)
    mse  = mean_squared_error(y_sev[test_idx], ys_hat)
    rmse = np.sqrt(mse)
    cm   = confusion_matrix(y_multi[test_idx], ym_hat, labels=np.arange(len(LABELS)))

    print("\n=== HOLD-OUT TEST METRICS ===")
    print(f"F1 (defect)         : {f1b:.3f}")
    print(f"F1 (macro, 7-class) : {f1m:.3f}")
    print(f"MAE(severity)       : {mae:.2f}")
    print(f"RMSE(severity)      : {rmse:.2f}")
    print("Confusion matrix (order):", LABELS)
    print(cm)

    # ---------- [4/4] Сохранение финальной модели (натренена на TRAIN) ----------
    print("[4/4] Сохранение...")
    joblib.dump(model, f"{MODEL_DIR}/mvp_model.joblib")
    with open(f"{MODEL_DIR}/features_info.json","w") as f:
        json.dump({"features_dim": int(X.shape[1]), "labels": LABELS, "train_files": train_files, "test_files": test_files}, f, ensure_ascii=False, indent=2)

    # Демонстрация TTF на одном тестовом файле (пример)
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
