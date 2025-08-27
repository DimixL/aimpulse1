# mvp_model_runtime.py
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

LABELS_ALL = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
TYPE2IDX   = {c:i for i,c in enumerate(LABELS_ALL)}
DEFECT_LABELS = ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]

@dataclass
class MVPModel:
    clf_bin: HistGradientBoostingClassifier
    calib_bin: object
    clf_def: HistGradientBoostingClassifier
    reg_sev: HistGradientBoostingRegressor
    scaler: StandardScaler
    label_map: dict
    thr_bin: float = 0.5
    tau_defclass: float = 0.30

    def predict_all(self, X: np.ndarray):
        Xs = self.scaler.transform(X)
        p_raw = np.clip(self.clf_bin.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
        p_def = self.calib_bin.predict(p_raw)
        y_bin_hat = (p_def >= self.thr_bin).astype(int)

        p_multi7   = np.zeros((Xs.shape[0], 7), dtype=float)
        y_multi_hat= np.full(Xs.shape[0], TYPE2IDX["normal"], dtype=int)
        sev_hat    = np.zeros(Xs.shape[0], dtype=float)

        idx = np.where(y_bin_hat == 1)[0]
        if len(idx) > 0:
            p_def6 = self.clf_def.predict_proba(Xs[idx])
            for j, name in enumerate(DEFECT_LABELS):
                p_multi7[idx, TYPE2IDX[name]] = p_def6[:, j]
            s = p_multi7[idx].sum(axis=1, keepdims=True) + 1e-12
            p_multi7[idx] = p_multi7[idx] / s

            conf = p_multi7[idx].max(axis=1)
            keep = conf >= self.tau_defclass
            idx_keep    = idx[keep]
            idx_reject  = idx[~keep]

            if len(idx_keep) > 0:
                y_multi_hat[idx_keep] = np.argmax(p_multi7[idx_keep], axis=1)
                sev_hat[idx_keep] = np.clip(self.reg_sev.predict(Xs[idx_keep]), 0, 100)
            if len(idx_reject) > 0:
                p_multi7[idx_reject, TYPE2IDX["normal"]] = 1.0

        p_multi7[y_bin_hat == 0, TYPE2IDX["normal"]] = 1.0
        return y_bin_hat, p_def, y_multi_hat, p_multi7, sev_hat
