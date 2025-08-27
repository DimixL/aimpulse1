# inference_v2.py
from __future__ import annotations
import os, io, json, joblib, warnings
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union
from functools import lru_cache
from scipy.signal import butter, filtfilt, hilbert, welch, decimate as sp_decimate, savgol_filter
import warnings

try:
    from features import extract_features_window as _extract_features_trained
except Exception as e:
    _extract_features_trained = None
    warnings.warn(f"features.extract_features_window не найден ({e}). "
                  f"Будет использован упрощённый фолбэк (31 фича) — это может не совпасть с моделью!")

# ==== мета ====
LABELS_ALL = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
TYPE2IDX = {c:i for i,c in enumerate(LABELS_ALL)}
IDX2TYPE = {i:c for c,i in TYPE2IDX.items()}
DEFECT_LABELS = ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]

COARSE_MAP = {
    TYPE2IDX["normal"]:        0,
    TYPE2IDX["BPFO"]:          1,
    TYPE2IDX["BPFI"]:          1,
    TYPE2IDX["BSF"]:           1,
    TYPE2IDX["FTF"]:           1,
    TYPE2IDX["imbalance"]:     2,
    TYPE2IDX["misalignment"]:  2,
}
COARSE_NAMES = ["normal", "bearing", "align/imbalance"]

# ==== доменные функции (точно как в тренинге) ====
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

# ==== фичи (как в тренинге) ====
def _fallback_extract_features_window(w: np.ndarray, fs: float) -> np.ndarray:
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

def extract_features_window(w: np.ndarray, fs: float) -> np.ndarray:
    # если есть «боевой» экстрактор из features.py — используем его
    if _extract_features_trained is not None:
        return _extract_features_trained(w, fs)
    # иначе — фолбэк
    return _fallback_extract_features_window(w, fs)

@dataclass
class FeatureCfg:
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

def read_csv_3phase(path_or_bytes: Union[str, io.BytesIO]) -> np.ndarray:
    if isinstance(path_or_bytes, (str, os.PathLike)):
        df = pd.read_csv(path_or_bytes)
    else:
        df = pd.read_csv(path_or_bytes)
    arr = df.iloc[:, :3].to_numpy(dtype=np.float32)  # A,B,C
    return arr

def sliding_windows(x: np.ndarray, fs: float, win_sec: float, overlap: float=0.0):
    n = len(x); wlen = int(win_sec*fs); step = int(wlen*(1-overlap))
    step = max(1, step)
    for i0 in range(0, max(0, n-wlen+1), step):
        yield i0, i0+wlen

def make_features_window(w: np.ndarray, fs: float, rpm: float, Z: int, dmm: float, Dmm: float, theta: int, cfg: FeatureCfg):
    f_base = extract_features_window(w, fs)
    lines = bearing_lines(rpm, Z, dmm, Dmm, theta)
    ph = choose_phase_for_env(w, fs, lines, cfg.mains_hz, cfg.mains_bw, cfg.fmax_env)
    f_env, P_env = envelope_psd(w[:, ph], fs, fmax=cfg.fmax_env, mains_hz=cfg.mains_hz, bw=cfg.mains_bw)

    def rel(fc, h=3):
        total = _area(f_env, P_env) + 1e-12
        return band_energy(f_env, P_env, fc, bw=1.0, harmonics=h) / total
    def sbr(fc):
        cen = band_energy(f_env, P_env, fc, bw=1.0, harmonics=1) + 1e-12
        sb1 = sideband_energy(f_env, P_env, fc, lines["fr"], k=1, bw=1.0)
        sb2 = sideband_energy(f_env, P_env, fc, lines["fr"], k=2, bw=1.0)
        return np.array([sb1/cen, sb2/cen], dtype=np.float32)

    f_dom = [rel(lines["FTF"],3), rel(lines["BPFO"],3), rel(lines["BPFI"],3), rel(lines["BSF"],3),
             rel(lines["fr"],2),  rel(2*lines["fr"],2),
             severity_from_bands(f_env, P_env, lines),
             defect_strength_ratio(f_env, P_env, lines)]
    for key in ["FTF","BPFO","BPFI","BSF"]:
        f_dom += list(sbr(lines[key]))
    feats = np.concatenate([f_base, np.array(f_dom, dtype=np.float32)], axis=0)
    return feats

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

# ==== основной класс ====
class PredictorV2:
    def __init__(self, model_dir: str = "models"):
        # --- shim для совместимости с пиклом "__main__.MVPModel" ---
        import sys
        from mvp_model_runtime import MVPModel as _MVPModel
        # отдадим распаковщику объект по тому имени, под которым он был сохранён
        sys.modules['__main__'].__dict__['MVPModel'] = _MVPModel
        # --- загружаем модель ---
        self.model = joblib.load(os.path.join(model_dir, "mvp_model_v2.joblib"))
        with open(os.path.join(model_dir, "features_info_v2.json"), "r") as f:
            self.info = json.load(f)
        self.cfg = FeatureCfg()
        self.labels7 = self.info.get("labels", LABELS_ALL)

    def predict_csv(self,
                    path_or_bytes: Union[str, io.BytesIO],
                    fs_raw: int = 25600,
                    fs_out: int = 3200,
                    win_sec: float = 1.0,
                    overlap: float = 0.0,
                    rpm: float = 1770.0,
                    Z: int = 9,
                    dmm: float = 7.94,
                    Dmm: float = 38.5,
                    theta: int = 0):
        x_raw = read_csv_3phase(path_or_bytes)
        x, fs = decimate_to_fs(x_raw, fs_raw, fs_out)

        feats, t0s, t1s = [], [], []
        for i0, i1 in sliding_windows(x, fs, win_sec, overlap):
            feats.append(make_features_window(x[i0:i1], fs, rpm, Z, dmm, Dmm, theta, self.cfg))
            t0s.append(i0 / fs); t1s.append(i1 / fs)
        if not feats:
            return pd.DataFrame(columns=["t0","t1","y_bin","p_def","y_pred","p7","severity"])

        X = np.vstack(feats).astype(np.float32)
        yb, pdef, y7, P7, sev = self.model.predict_all(X)

        # свёртка в 3 класса
        P3 = np.zeros((P7.shape[0], 3), dtype=float)
        P3[:,0] = P7[:, TYPE2IDX["normal"]]
        P3[:,1] = P7[:, TYPE2IDX["BPFO"]] + P7[:, TYPE2IDX["BPFI"]] + P7[:, TYPE2IDX["BSF"]] + P7[:, TYPE2IDX["FTF"]]
        P3[:,2] = P7[:, TYPE2IDX["imbalance"]] + P7[:, TYPE2IDX["misalignment"]]

        df = pd.DataFrame({
            "t0": t0s, "t1": t1s,
            "y_bin": yb.astype(int),
            "p_def": pdef.astype(float),
            "y_pred": [IDX2TYPE[i] for i in y7],
            "severity": sev.astype(float),
            "p7": list(P7.astype(float)),
            "p3": list(P3.astype(float)),
        })

        # агрегаты по файлу
        p7_mean = P7.mean(axis=0)
        p3_mean = P3.mean(axis=0)
        file_pred7 = int(np.argmax(p7_mean))
        file_pred3 = int(np.argmax(p3_mean))

        agg = {
            "p_def_mean": float(pdef.mean()),
            "p_def_share_over_thr": float((pdef >= float(self.info.get("bin_threshold", 0.5))).mean()),
            "file_class7": IDX2TYPE[file_pred7],
            "file_p7": p7_mean.tolist(),
            "file_class3": COARSE_NAMES[file_pred3],
            "file_p3": p3_mean.tolist(),
            "severity_mean": float(sev.mean()),
            "severity_max": float(sev.max() if len(sev) else 0.0),
            "ttf_to_80_sec": float(forecast_ttf_for_file(np.clip(sev,0,100), np.array(t1s), 80.0)),
            "thr_bin": float(self.info.get("bin_threshold", 0.5)),
        }
        return df, agg

# удобный враппер для Streamlit
_predictor_singleton = None
def load_predictor(model_dir="models"):
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = PredictorV2(model_dir)
    return _predictor_singleton
