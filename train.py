# train.py
import os, json, glob, numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from features import read_csv_3phase, sliding_windows, extract_features_window, decimate
from models import MLPClassifier, MLPRegressor, save_bundle, N_CLASSES
import joblib
from sklearn.ensemble import IsolationForest

FS_RAW   = 25600
FS_OUT   = 5120       # можно 3200 при желании
WIN_SEC  = 1.0
OVERLAP  = 0.5

def train_oneclass(X):
    """Unsupervised: учим 'норму' и ищем аномалии."""
    os.makedirs("artifacts", exist_ok=True)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    oc = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    oc.fit(Xs)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(oc,     "artifacts/oc_iforest.joblib")
    json.dump({"mode":"oneclass"}, open("artifacts/meta.json","w"))
    print("✓ Saved one-class model → artifacts/")
    return scaler, oc

if __name__=="__main__":
    files = sorted(glob.glob("data/raw/*.csv"))
    labels = pd.read_csv("data/labels.csv") if os.path.exists("data/labels.csv") else None
    X, yb, ym, ys, groups = build_dataset(files, labels)

    # если меток нет или нет «дефектов» → учим one-class
    if labels is None or labels.empty or (len([v for v in yb if v==1]) < 5):
        train_oneclass(X)
    else:
        scaler, cls, reg = train_supervised(X, yb, ym, ys, groups)
        save_bundle(scaler, cls, reg, out_dir="artifacts")

def build_dataset(files, labels_df):
    X, y_bin, y_mc, y_sev, groups = [], [], [], [], []
    for f in files:
        x = read_csv_3phase(f)
        factor = max(1, FS_RAW // FS_OUT)
        if factor>1: x = decimate(x, factor)
        fs = FS_RAW // factor
        for s, win in sliding_windows(x, fs, WIN_SEC, OVERLAP):
            feat = extract_features_window(win, fs)   # 90 фичей это 30 на 30 если чтооооо
            X.append(feat); groups.append(os.path.basename(f))
            if labels_df is not None and not labels_df.empty:
                # появилась ли метка в этом временном окне
                rel_t = s/fs
                lab = labels_df[(labels_df.file==os.path.basename(f)) &
                                (labels_df.ts_start<=rel_t) & (labels_df.ts_end>rel_t)]
                if len(lab):
                    row = lab.iloc[0]
                    fault = row['fault_type']
                    if fault=='healthy':
                        y_bin.append(0); y_mc.append(0); y_sev.append(0.0)
                    else:
                        y_bin.append(1)
                        # кодировка классов: 1..6 по порядку типов
                        cls_map = {'bearing_outer':1,'bearing_inner':2,'rolling':3,'cage':4,'imbalance':5,'misalignment':6}
                        y_mc.append(cls_map[fault])
                        y_sev.append(float(row.get('severity',50)))
                else:
                    # нет метки — помечаем как неизвестно (пропустим при обучении)
                    y_bin.append(None); y_mc.append(None); y_sev.append(None)
            else:
                y_bin.append(None); y_mc.append(None); y_sev.append(None)
    X = np.array(X, np.float32); groups = np.array(groups)
    return X, np.array(y_bin, object), np.array(y_mc, object), np.array(y_sev, object), groups

def train_supervised(X, y_bin, y_mc, y_sev, groups):
    # оставляем только размеченные окна
    mask = np.array([v is not None for v in y_bin])
    X, y_bin, y_mc, y_sev, groups = X[mask], y_bin[mask].astype(int), y_mc[mask].astype(int), y_sev[mask].astype(float), groups[mask]
    # групповой сплит по файлам
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y_bin, groups))
    scaler = StandardScaler().fit(X[tr]); Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
    # модели
    device = torch.device('cpu')
    cls, reg = MLPClassifier().to(device), MLPRegressor().to(device)
    opt_c = torch.optim.Adam(cls.parameters(), 1e-3); opt_r = torch.optim.Adam(reg.parameters(), 1e-3)
    bce = torch.nn.BCEWithLogitsLoss(); ce = torch.nn.CrossEntropyLoss(); mse = torch.nn.MSELoss()

    def train_epoch():
        cls.train(); reg.train()
        xb = torch.tensor(Xtr, dtype=torch.float32)
        yb_bin = torch.tensor(y_bin[tr], dtype=torch.float32).unsqueeze(1)
        yb_mc  = torch.tensor(y_mc[tr],  dtype=torch.long)
        yb_sev = torch.tensor(y_sev[tr], dtype=torch.float32).unsqueeze(1)
        # классификатор
        opt_c.zero_grad()
        logit_bin, logit_mc = cls(xb)
        loss_c = bce(logit_bin, yb_bin) + ce(logit_mc, yb_mc)
        loss_c.backward(); opt_c.step()
        # регрессор
        opt_r.zero_grad()
        xr = xb; pr = reg(xr)
        loss_r = mse(pr, yb_sev)
        loss_r.backward(); opt_r.step()
        return float(loss_c.item()), float(loss_r.item())

    for epoch in range(40):
        lc, lr = train_epoch()
        if epoch%5==0: print(f"epoch {epoch}: cls {lc:.3f} reg {lr:.3f}")

    # простая оценка
    cls.eval(); reg.eval()
    xt = torch.tensor(Xte, dtype=torch.float32)
    pb = torch.sigmoid(cls(xt)[0]).detach().numpy().ravel()
    mc = torch.softmax(cls(xt)[1], dim=1).detach().numpy()
    se = reg(xt).detach().numpy().ravel()
    print("valid: bin-acc:", ((pb>0.5)==y_bin[te]).mean())

    return scaler, cls, reg

if __name__=="__main__":
    files = sorted(glob.glob("data/raw/*.csv"))
    labels = pd.read_csv("data/labels.csv") if os.path.exists("data/labels.csv") else None
    X, yb, ym, ys, groups = build_dataset(files, labels)
    scaler, cls, reg = train_supervised(X, yb, ym, ys, groups)
    save_bundle(scaler, cls, reg, out_dir="artifacts")
