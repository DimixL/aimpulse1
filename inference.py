import numpy as np, joblib, torch, json
from features import read_csv_3phase, decimate, sliding_windows, extract_features_window
from models import MLPClassifier, MLPRegressor
import os

def load_bundle(path="artifacts"):
    meta_path = os.path.join(path, "meta.json")
    mode = "supervised"
    if os.path.exists(meta_path):
        try:
            mode = json.load(open(meta_path))["mode"]
        except Exception:
            pass
    if mode == "oneclass" or not os.path.exists(os.path.join(path,"cls_state.pt")):
        return {"mode":"oneclass",
                "scaler": joblib.load(f"{path}/scaler.pkl"),
                "oc":     joblib.load(f"{path}/oc_iforest.joblib")}
    # supervised
    scaler = joblib.load(f"{path}/scaler.pkl")
    cls = MLPClassifier(); cls.load_state_dict(torch.load(f"{path}/cls_state.pt", map_location="cpu")); cls.eval()
    reg = MLPRegressor(); reg.load_state_dict(torch.load(f"{path}/reg_state.pt", map_location="cpu")); reg.eval()
    return {"mode":"supervised", "scaler":scaler, "cls":cls, "reg":reg}

def _severity_from_anomaly(scores):
    s = (scores - np.percentile(scores, 10)) / (np.std(scores)+1e-9)
    return np.clip(50 + 25*s, 0, 100)  # 0..100

def predict_csv(csv_path, fs_raw=25600, fs_out=5120, win_sec=1.0, overlap=0.0):
    x = read_csv_3phase(csv_path)
    factor = max(1, fs_raw//fs_out)
    if factor>1: x = decimate(x, factor)
    fs = fs_raw//factor
    bundle = load_bundle()
    results, feats_all = [], []

    # сначала посчитаем фичи для всех окон (нужно для нормализации severity в one-class)
    wins = list(sliding_windows(x, fs, win_sec, overlap))
    for start, win in wins:
        feats_all.append(extract_features_window(win, fs))
    X = np.array(feats_all, np.float32)
    Xs = bundle["scaler"].transform(X)

    if bundle["mode"] == "oneclass":
        oc = bundle["oc"]
        # чем меньше decision_function, тем «аномальнее»
        d = -oc.decision_function(Xs)               # >0 → аномалия
        p = 1/(1+np.exp(-4*(d - d.mean())))         # псевдо-вероятность
        sev_all = _severity_from_anomaly(d)
        for (start,_), p_fault, sev in zip(wins, p, sev_all):
            results.append({
                "t0": start/fs, "t1": (start+int(win_sec*fs))/fs,
                "is_fault": bool(p_fault>0.5), "p_fault": float(p_fault),
                "proba": [0,0,0,0,0,0],               # тип неизвестен в unsup
                "severity": float(sev)
            })
        return results

    # supervised
    cls, reg = bundle["cls"], bundle["reg"]
    xt = torch.tensor(Xs, dtype=torch.float32)
    p_bin = torch.sigmoid(cls(xt)[0]).detach().numpy().ravel()
    p_mc  = torch.softmax(cls(xt)[1], dim=1).detach().numpy()
    sev   = reg(xt).detach().numpy().ravel()
    for (start,_), pb, pmc, sv in zip(wins, p_bin, p_mc, sev):
        results.append({
            "t0": start/fs, "t1": (start+int(win_sec*fs))/fs,
            "is_fault": bool(pb>0.5), "p_fault": float(pb),
            "proba": pmc.tolist(),
            "severity": float(np.clip(sv,0,100))
        })
    return results
