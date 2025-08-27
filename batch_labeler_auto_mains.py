#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, time, datetime as dt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from scipy.signal import welch, butter, filtfilt, hilbert

# ---- импорт твоих утилит из features.py ----
from features import read_csv_3phase, decimate, sliding_windows, extract_features_window

# --------------------- базовые параметры ---------------------
FS_RAW   = 25600
FS_OUT   = 3200
WIN_SEC  = 1.0
OVERLAP  = 0.5

LABELS   = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]

def bearing_lines(rpm, Z, d_mm, D_mm, theta_deg):
    fr = rpm / 60.0
    theta = np.deg2rad(theta_deg)
    r = (d_mm / D_mm) * np.cos(theta)
    FTF  = 0.5 * fr * (1 - r)
    BPFO = 0.5 * Z * fr * (1 - r)
    BPFI = 0.5 * Z * fr * (1 + r)
    BSF  = (D_mm / d_mm) * fr * 0.5 * (1 - r**2)
    return dict(fr=fr, FTF=FTF, BPFO=BPFO, BPFI=BPFI, BSF=BSF)

def _area(f, P, m=None):
    if m is None: return float(np.trapezoid(P, f))
    return float(np.trapezoid(P[m], f[m]))

def envelope_psd(y, fs, fmax=320.0, mains_hz=60.0, bw=10.0):
    """БПФ огибающей после полосового фильтра вокруг сети."""
    y = y - y.mean()
    lo = max(1.0, mains_hz - bw) / (fs/2)
    hi = min(fs/2 - 1.0, mains_hz + bw) / (fs/2)
    b, a = butter(4, [lo, hi], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, y)))
    f, P = welch(env, fs=fs, nperseg=min(4096, len(env)))
    m = f <= fmax
    return f[m], P[m]

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
    denom = _area(f,P) + 1e-12
    return float(np.clip(400.0 * numer/denom, 0, 100))

def choose_phase_for_env(w, fs, lines, mains_hz, bw, fmax):
    best_idx, best_val = 0, -1.0
    for idx in range(min(w.shape[1], 3)):
        f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mains_hz, bw=bw)
        val = sum(band_energy(f,P,lines[k]) for k in ["BPFO","BPFI","BSF","FTF"])
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx

def compute_window_scores(y, fs, lines, mains_hz, bw, fmax):
    f, P = envelope_psd(y, fs, fmax=fmax, mains_hz=mains_hz, bw=bw)
    total = _area(f, P) + 1e-12
    get = lambda fc, h=3: band_energy(f, P, fc, bw=1.0, harmonics=h)
    scores = {
        "BPFO": get(lines["BPFO"], 3)/total,
        "BPFI": get(lines["BPFI"], 3)/total,
        "BSF":  get(lines["BSF"],  3)/total,
        "FTF":  get(lines["FTF"],  3)/total,
        "imbalance":    get(lines["fr"],      2)/total,
        "misalignment": get(2.0*lines["fr"], 2)/total,
    }
    sev = severity_from_bands(f, P, lines)
    defect_strength = scores["BPFO"] + scores["BPFI"] + scores["BSF"] + scores["FTF"]
    return sev, scores, defect_strength

# -------------------- автоопределение сети 50/60 --------------------
def detect_mains_psd(x, fs, bw=1.0):
    """Возвращает 50.0 или 60.0 — где больше энергия в ±bw Гц, усредняя по фазам."""
    f, Psum = None, None
    for k in range(min(x.shape[1], 3)):
        f_k, P_k = welch(x[:,k] - x[:,k].mean(), fs=fs, nperseg=min(8192, len(x)))
        if f is None:
            f, Psum = f_k, P_k
        else:
            Psum = Psum + P_k
    def band_en(fc):
        m = (f >= fc-bw) & (f <= fc+bw)
        return float(Psum[m].sum())
    e50, e60 = band_en(50.0), band_en(60.0)
    return (50.0 if e50 >= e60 else 60.0), e50, e60

def detect_mains_by_defect_energy(w, fs, lines, fmax, mains_bw=10.0):
    """Если PSD не дал ясного ответа — берём тот вариант сети,
    при котором сумма «дефектных» энергий выше."""
    scores = {}
    for mhz in (50.0, 60.0):
        idx = choose_phase_for_env(w, fs, lines, mhz, mains_bw, fmax)
        f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mhz, bw=mains_bw)
        scores[mhz] = sum(band_energy(f,P,lines[k]) for k in ["BPFO","BPFI","BSF","FTF"])
    return 50.0 if scores[50.0] >= scores[60.0] else 60.0

# -------------------- пайплайн по файлу --------------------
def process_one_file(path, args):
    # чтение и децимация
    x = read_csv_3phase(path)   # Nx3
    factor = max(1, FS_RAW // args.fs_out)
    x2 = decimate(x, factor) if factor > 1 else x
    fs = FS_RAW // factor
    lines_nominal = bearing_lines(args.rpm, args.Z, args.dmm, args.Dmm, args.theta)

    # авто-детект сети
    mains_hz, e50, e60 = detect_mains_psd(x2, fs, bw=1.0)
    if abs(e50 - e60) / (max(e50, e60) + 1e-12) < 0.15:
        # слабое различие → проверим по «дефектной энергии» на одном окне
        L = int(args.win * fs)
        s0 = 0 if len(x2) < L else (len(x2)//2 - L//2)
        mains_hz = detect_mains_by_defect_energy(x2[s0:s0+L], fs, lines_nominal, args.fmax, args.mains_bw)

    # скользящие окна
    spans = [(s, s+int(args.win*fs)) for s,_ in sliding_windows(x2, fs, args.win, args.overlap)]

    # посчитаем «фичи» для аномалий + быстрые метрики для типов
    feats, severities, strengths, type_scores = [], [], [], []
    for s, w in sliding_windows(x2, fs, args.win, args.overlap):
        feats.append(extract_features_window(w, fs))
        idx = choose_phase_for_env(w, fs, lines_nominal, mains_hz, args.mains_bw, args.fmax)
        sev, sc, strg = compute_window_scores(w[:, idx], fs, lines_nominal, mains_hz, args.mains_bw, args.fmax)
        severities.append(sev); strengths.append(strg); type_scores.append(sc)

    X = np.array(feats, np.float32)
    severities  = np.array(severities, np.float32)
    strengths   = np.array(strengths,  np.float32)

    # ---------- авто-норма ----------
    normals_idx = []
    if len(X) >= 5:
        oc = IsolationForest(n_estimators=300, contamination=args.contam, random_state=42).fit(X)
        anom = -oc.decision_function(X)
    else:
        anom = np.zeros(len(X), dtype=float)

    mask_calm = (severities <= args.sev_thr) & (strengths <= args.strength_thr)
    idx_pool  = np.where(mask_calm)[0]
    if idx_pool.size:
        order = idx_pool[np.argsort(anom[idx_pool])]
        normals_idx = list(order[:min(args.n_norm, len(order))])

    # ---------- кандидаты по типам ----------
    buckets = {lab: [] for lab in ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]}
    for j, sc in enumerate(type_scores):
        for lab in buckets:
            buckets[lab].append((sc[lab], j))

    type_rows = []
    for lab, arr in buckets.items():
        arr = [a for a in arr if a[0] >= args.min_score]
        arr.sort(key=lambda x: -x[0])
        for score, j in arr[:args.topk]:
            s0, s1 = spans[j]
            type_rows.append(dict(
                win_id=f"{os.path.basename(path)}:{s0}:{s1}",
                file=os.path.basename(path), i0=s0, i1=s1, fs=fs,
                t0=s0/fs, t1=s1/fs, y_defect=int(lab!="normal"),
                y_type=lab, severity=float(severities[j]),
                confidence=float(np.clip(score/max(args.min_score,1e-6), 0.3, 1.0)),
                notes=f"auto-{lab} score={score:.4f}",
                rpm=float(args.rpm), Z=int(args.Z), dmm=float(args.dmm),
                Dmm=float(args.Dmm), theta=int(args.theta),
                mains_hz=float(mains_hz)
            ))

    # ---------- собрать «нормы» ----------
    norm_rows = []
    for j in normals_idx:
        s0, s1 = spans[j]
        norm_rows.append(dict(
            win_id=f"{os.path.basename(path)}:{s0}:{s1}",
            file=os.path.basename(path), i0=s0, i1=s1, fs=fs,
            t0=s0/fs, t1=s1/fs, y_defect=0, y_type="normal",
            severity=float(severities[j]), confidence=0.5,
            notes="auto-normal",
            rpm=float(args.rpm), Z=int(args.Z), dmm=float(args.dmm),
            Dmm=float(args.Dmm), theta=int(args.theta),
            mains_hz=float(mains_hz)
        ))

    return norm_rows + type_rows, mains_hz

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser("Batch labeling with mains auto-detection (50/60 Hz)")
    p.add_argument("--data-glob", required=True, help="Путь-шаблон до CSV, напр. data/raw/*.csv")
    p.add_argument("--out-dir",    required=True, help="Куда писать labels_*.csv")
    # bearing
    p.add_argument("--rpm", type=float, default=1770.0)
    p.add_argument("--Z",   type=int,   default=9)
    p.add_argument("--dmm", type=float, default=7.94)
    p.add_argument("--Dmm", type=float, default=38.5)
    p.add_argument("--theta", type=float, default=0.0)
    # signal / demod
    p.add_argument("--fs-out", type=int, default=FS_OUT)
    p.add_argument("--win", type=float, default=WIN_SEC)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    p.add_argument("--fmax", type=float, default=320.0)
    p.add_argument("--mains-bw", type=float, default=10.0)
    # auto-normal
    p.add_argument("--n-norm", type=int, default=15)
    p.add_argument("--sev-thr", type=float, default=10.0)
    p.add_argument("--strength-thr", type=float, default=0.05)
    p.add_argument("--contam", type=float, default=0.05)
    # candidates per type
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--min-score", type=float, default=5e-4)  # мягче по умолчанию
    return p.parse_args()

def main():
    args = parse_args()
    files = sorted(glob.glob(args.data_glob))
    assert len(files) > 0, f"Файлы не найдены: {args.data_glob}"

    t0 = time.time()
    all_rows = []
    mains_stat = {50.0:0, 60.0:0}

    for p in files:
        rows, mains_hz = process_one_file(p, args)
        all_rows.extend(rows)
        mains_stat[50.0 if abs(mains_hz-50.0) < 0.6 else 60.0] += 1
        print(f"[{os.path.basename(p)}] mains≈{mains_hz:.1f} Hz  -> labels {len(rows)}")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["win_id"], keep="last")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"labels_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
    df.to_csv(out, index=False)

    counts = df["y_type"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print("\n================ SUMMARY ================")
    print(f"Labels written: {len(df)}  ->  {out}")
    for k in LABELS:
        print(f"{k:<12}: {int(counts.get(k,0))}")
    print(f"Processed files: {len(files)}   Time: {time.time()-t0:.1f}s")
    print(f"Mains chosen per-file -> 50Hz: {mains_stat[50.0]} | 60Hz: {mains_stat[60.0]}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
