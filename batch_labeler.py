#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch auto-labeler for AImpulse dataset (no UI).
- Scans CSVs, finds 6 defect types + normal
- Writes labels_{timestamp}.csv with one label per window
- Prints dataset-wide stats

Assumes 'features.py' provides:
  - read_csv_3phase(path) -> Nx3 float array
  - decimate(x, factor)   -> Nx3 float array
  - sliding_windows(x, fs, win_sec, overlap) -> iterator of (start_index, window_array_Nx3)
  - extract_features_window(w, fs) -> 1D feature vector (for IsolationForest)

Author: you, but calmer :)
"""

import os
import glob
import argparse
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.signal import butter, filtfilt, hilbert, welch

# ---- your project utils ----
from features import read_csv_3phase, decimate, sliding_windows, extract_features_window

# -------------------- Defaults (tweak as needed) --------------------
FS_RAW     = 25600
FS_OUT     = 3200
WIN_SEC    = 1.0
OVERLAP    = 0.5

# Bearing / kinematics (your defaults)
RPM        = 1770.0
Z          = 9
D_MM       = 38.5
d_MM       = 7.94
THETA_DEG  = 0.0

# Envelope / demod
MAINS_HZ   = 50.0     # 50 or 60
MAINS_BW   = 10.0
FMAX_ENV   = 320.0

# Normal windows selection
NORM_PER_FILE   = 15
SEV_THR         = 10.0     # max severity for normal
STR_THR         = 0.05     # max sum of defect-band energy share
IF_CONTAM       = 0.05     # IsolationForest

# Defect candidates selection
TYPES           = ("BPFO","BPFI","BSF","FTF","imbalance","misalignment")
TOPK_PER_TYPE   = 30
MIN_SCORE       = 1e-3     # increase for higher precision, decrease for more recall

RNG_SEED        = 42

# -------------------- Helpers --------------------
def bearing_lines(rpm, Z, d_mm, D_mm, theta_deg):
    fr = rpm / 60.0
    theta = np.deg2rad(theta_deg)
    r = (d_mm / D_mm) * np.cos(theta)
    FTF  = 0.5 * fr * (1 - r)
    BPFO = 0.5 * Z * fr * (1 - r)
    BPFI = 0.5 * Z * fr * (1 + r)
    BSF  = (D_mm / d_mm) * fr * 0.5 * (1 - r**2)
    return dict(fr=fr, FTF=FTF, BPFO=BPFO, BPFI=BPFI, BSF=BSF)

def envelope_psd(y, fs, fmax=FMAX_ENV, mains_hz=MAINS_HZ, bw=MAINS_BW):
    """Bandpass around mains±bw, Hilbert envelope, Welch PSD up to fmax."""
    if y.ndim != 1:
        y = y.ravel()
    y = y - y.mean()
    lo = max(1.0, mains_hz - bw) / (fs/2)
    hi = min(fs/2 - 1.0, mains_hz + bw) / (fs/2)
    lo = np.clip(lo, 1e-6, 0.99); hi = np.clip(hi, lo + 1e-6, 0.999)
    b, a = butter(4, [lo, hi], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, y)))
    f, P = welch(env, fs=fs, nperseg=min(4096, len(env)))
    m = f <= fmax
    return f[m], P[m]

def _area(f, P, m=None):
    if m is None:
        return float(np.trapezoid(P, f))
    return float(np.trapezoid(P[m], f[m]))

def band_energy(f, P, fc, bw=2.0, harmonics=1):
    """Integrate energy around fc and its harmonics."""
    if fc <= 0:
        return 0.0
    e = 0.0
    for k in range(1, harmonics+1):
        fck = fc * k
        bwk = max(bw, 0.02 * fck)  # at least ±1 Hz or 2%
        m = (f >= fck - bwk) & (f <= fck + bwk)
        if m.any():
            e += _area(f, P, m)
    return e

def choose_phase_for_env(w, fs, lines, mains_hz, bw, fmax):
    """Pick A/B/C phase with max summed bearing-band energy."""
    best_idx, best_val = 0, -1.0
    for idx in range(min(w.shape[1], 3)):
        f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mains_hz, bw=bw)
        val = sum(band_energy(f, P, lines[k]) for k in ["BPFO","BPFI","BSF","FTF"])
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx

def compute_window_scores(y, fs, lines, mains_hz, bw, fmax):
    """Return severity (0..100), per-type relative energies, and total defect strength."""
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
    # severity = share of bearing defect bands energy (scaled to 0..100)
    numer = (get(lines["BPFO"], 3) + get(lines["BPFI"], 3) + get(lines["BSF"], 3) + get(lines["FTF"], 3))
    sev = float(np.clip(400.0 * numer/total, 0, 100))
    defect_strength = scores["BPFO"] + scores["BPFI"] + scores["BSF"] + scores["FTF"]
    return sev, scores, defect_strength

def load_and_resample(path, fs_out=FS_OUT):
    x = read_csv_3phase(path)     # Nx3
    factor = max(1, int(round(FS_RAW / fs_out)))
    if factor > 1:
        x2 = decimate(x, factor)
        fs = FS_RAW // factor
    else:
        x2 = x
        fs = FS_RAW
    return x2.astype(np.float32), int(fs)

def window_id(path, i0, i1):
    return f"{os.path.basename(path)}:{i0}:{i1}"

# -------------------- Core pipeline --------------------
def scan_file(path, *,
              fs_out=FS_OUT, win_sec=WIN_SEC, overlap=OVERLAP,
              rpm=RPM, Z=Z, dmm=d_MM, Dmm=D_MM, theta=THETA_DEG,
              mains_hz=MAINS_HZ, mains_bw=MAINS_BW, fmax=FMAX_ENV):
    """
    For each window:
      - pick phase
      - compute severity, per-type scores, defect_strength
      - also extract generic features for IF
    Returns:
      spans [(i0,i1)], severities, defect_strengths, type_scores(list of dict), features(array)
    """
    x, fs = load_and_resample(path, fs_out)
    L = int(win_sec * fs)
    lines = bearing_lines(rpm, Z, dmm, Dmm, theta)

    spans, sevs, strengths, type_scores, feats = [], [], [], [], []
    for i0, w in sliding_windows(x, fs, win_sec, overlap):
        i1 = i0 + L
        spans.append((i0, i1))
        feats.append(extract_features_window(w, fs))
        idx = choose_phase_for_env(w, fs, lines, mains_hz, mains_bw, fmax)
        sev, sc, strg = compute_window_scores(w[:, idx], fs, lines, mains_hz, mains_bw, fmax)
        sevs.append(sev); strengths.append(strg); type_scores.append(sc)
    return (x, fs, spans, np.array(sevs, np.float32),
            np.array(strengths, np.float32), type_scores, np.array(feats, np.float32))

def select_normals(spans, feats, sevs, strengths,
                   *, n_per_file=NORM_PER_FILE, sev_thr=SEV_THR,
                   str_thr=STR_THR, if_contam=IF_CONTAM):
    """Pick calm windows (low severity & low defect-strength), then lowest anomaly by IF."""
    idx_pool = np.where((sevs <= sev_thr) & (strengths <= str_thr))[0]
    if len(idx_pool) == 0:
        return []

    if feats.shape[0] >= 5:
        oc = IsolationForest(n_estimators=300, contamination=if_contam,
                             random_state=RNG_SEED).fit(feats)
        anom = -oc.decision_function(feats)  # higher -> more anomalous
    else:
        anom = np.zeros(len(spans), dtype=float)

    order = idx_pool[np.argsort(anom[idx_pool])]  # calmest first
    return list(order[:min(n_per_file, len(order))])

def select_defects(type_scores, *, labels=TYPES, topk=TOPK_PER_TYPE, min_score=MIN_SCORE):
    """
    Build candidate lists per label, keep topk per label by score.
    Returns dict: label -> [(score, idx), ...]
    """
    buckets = {lab: [] for lab in labels}
    for idx, sc in enumerate(type_scores):
        for lab in labels:
            score = float(sc[lab])
            if score >= min_score:
                buckets[lab].append((score, idx))
    for lab in buckets:
        buckets[lab].sort(key=lambda t: -t[0])
        buckets[lab] = buckets[lab][:topk]
    return buckets

def fuse_labels(spans, fs, sevs, type_scores, buckets, normal_idxs,
                *, rpm=RPM, Z=Z, dmm=d_MM, Dmm=D_MM, theta=THETA_DEG,
                min_score=MIN_SCORE):
    """
    One label per window: if a window appears as defect in several buckets,
    keep the label with the highest score. Defects override normals.
    Returns list of dict label records.
    """
    records = {}
    # 1) defects: pick best label per window
    for lab, arr in buckets.items():
        for score, j in arr:
            # dominance over 2nd best class (except self)
            scj = type_scores[j]
            best_other = max(float(scj[k]) for k in scj.keys() if k != lab)
            dom = score / (best_other + 1e-9)
            # confidence: (how far above threshold) + dominance
            conf = np.clip(0.5*((score - min_score)/(4*min_score + 1e-12)) + 0.5*np.clip(dom-1, 0, 1), 0.0, 1.0)

            i0, i1 = spans[j]
            rec = dict(
                win_id=window_id("<many files>", i0, i1),  # replaced later per-file
                file="", i0=i0, i1=i1, t0=i0/fs, t1=i1/fs, fs=fs,
                y_defect=1, y_type=lab, severity=float(sevs[j]),
                confidence=float(conf), notes=f"auto-{lab} score={score:.4f}",
                rpm=float(rpm), Z=int(Z), dmm=float(dmm), Dmm=float(Dmm), theta=int(theta)
            )
            # If window already has a defect label, keep the stronger one
            if j not in records or score > records[j]["_score"]:
                rec["_score"] = score
                records[j] = rec

    # 2) normals: add only if this window is not defect-labeled
    for j in normal_idxs:
        if j in records:
            continue
        i0, i1 = spans[j]
        rec = dict(
            win_id=window_id("<many files>", i0, i1),
            file="", i0=i0, i1=i1, t0=i0/fs, t1=i1/fs, fs=fs,
            y_defect=0, y_type="normal", severity=float(sevs[j]),
            confidence=0.5, notes="auto-normal",
            rpm=float(rpm), Z=int(Z), dmm=float(dmm), Dmm=float(Dmm), theta=int(theta),
            _score=0.0
        )
        records[j] = rec

    # return ordered by window index
    return [records[k] for k in sorted(records.keys())]

# -------------------- Runner --------------------
def run(args):
    np.random.seed(RNG_SEED)

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f"No CSV files matched: {args.data_glob}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"labels_{ts}.csv")

    all_records = []
    per_type_counts = defaultdict(int)

    t0 = time.time()
    for path in files:
        try:
            x, fs, spans, sevs, strengths, type_scores, feats = scan_file(
                path,
                fs_out=args.fs_out, win_sec=args.win, overlap=args.overlap,
                rpm=args.rpm, Z=args.z, dmm=args.dmm, Dmm=args.Dmm, theta=args.theta,
                mains_hz=args.mains_hz, mains_bw=args.mains_bw, fmax=args.fmax
            )

            normal_idxs = select_normals(
                spans, feats, sevs, strengths,
                n_per_file=args.norm_per_file, sev_thr=args.sev_thr,
                str_thr=args.str_thr, if_contam=args.if_contam
            )

            buckets = select_defects(
                type_scores, labels=TYPES, topk=args.topk, min_score=args.min_score
            )

            fused = fuse_labels(
                spans, fs, sevs, type_scores, buckets, normal_idxs,
                rpm=args.rpm, Z=args.z, dmm=args.dmm, Dmm=args.Dmm, theta=args.theta,
                min_score=args.min_score
            )

            # fix file name & win_id per-file; collect stats
            for r in fused:
                r["file"] = os.path.basename(path)
                r["win_id"] = f"{r['file']}:{r['i0']}:{r['i1']}"
                per_type_counts[r["y_type"]] += 1

            all_records.extend(fused)

            print(f"[OK] {os.path.basename(path)}: "
                  f"{sum(1 for r in fused if r['y_type']!='normal')} defects, "
                  f"{sum(1 for r in fused if r['y_type']=='normal')} normals "
                  f"(windows={len(spans)})")
        except Exception as e:
            print(f"[WARN] {os.path.basename(path)} skipped: {e}")

    if not all_records:
        print("[DONE] No labels produced.")
        return

    df = pd.DataFrame(all_records)
    df = df[[
        "win_id","file","i0","i1","t0","t1","fs",
        "y_defect","y_type","severity","confidence","notes",
        "rpm","Z","dmm","Dmm","theta"
    ]]
    df.to_csv(out_path, index=False)

    dt = time.time() - t0
    total = len(df)
    print("\n================ SUMMARY ================")
    print(f"Labels written: {total}  ->  {out_path}")
    for lab in ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]:
        print(f"{lab:12s}: {per_type_counts.get(lab,0)}")
    print(f"Processed files: {len(files)}   Time: {dt:.1f}s")
    print("=========================================\n")

# -------------------- CLI --------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Batch auto-labeler (no UI).")
    p.add_argument("--data-glob", default="data/raw/*.csv",
                   help="Glob for input CSVs")
    p.add_argument("--out-dir", default="labels",
                   help="Where to store labels_*.csv")
    # processing
    p.add_argument("--fs-out", type=int, default=FS_OUT)
    p.add_argument("--win", type=float, default=WIN_SEC)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    # kinematics
    p.add_argument("--rpm", type=float, default=RPM)
    p.add_argument("--z", type=int, default=Z)
    p.add_argument("--dmm", type=float, default=d_MM)
    p.add_argument("--Dmm", type=float, default=D_MM)
    p.add_argument("--theta", type=float, default=THETA_DEG)
    # demod
    p.add_argument("--mains-hz", type=float, default=MAINS_HZ)
    p.add_argument("--mains-bw", type=float, default=MAINS_BW)
    p.add_argument("--fmax", type=float, default=FMAX_ENV)
    # normals
    p.add_argument("--norm-per-file", type=int, default=NORM_PER_FILE)
    p.add_argument("--sev-thr", type=float, default=SEV_THR)
    p.add_argument("--str-thr", type=float, default=STR_THR)
    p.add_argument("--if-contam", type=float, default=IF_CONTAM)
    # defects
    p.add_argument("--topk", type=int, default=TOPK_PER_TYPE)
    p.add_argument("--min-score", type=float, default=MIN_SCORE)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
