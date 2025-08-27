# models.py
import torch, torch.nn as nn, numpy as np, joblib, json, os
from sklearn.preprocessing import StandardScaler

N_FEATS = 90
N_CLASSES = 6

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(N_FEATS, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_bin = nn.Linear(32, 1)          # y1
        self.head_mc  = nn.Linear(32, N_CLASSES)  # y2
    def forward(self, x):
        h = self.backbone(x)
        return self.head_bin(h), self.head_mc(h)

class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATS, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

def save_bundle(scaler, cls_model, reg_model, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    torch.save(cls_model.state_dict(), f"{out_dir}/cls_state.pt")
    torch.save(reg_model.state_dict(), f"{out_dir}/reg_state.pt")
    json.dump({"n_feats":N_FEATS}, open(f"{out_dir}/feature_schema.json","w"))