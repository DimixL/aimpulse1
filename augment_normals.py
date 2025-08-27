#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_normals.py — добавляем только 'normal' в метки и пишем новый labels-файл.

Пример:
  python augment_normals.py \
      --labels-in labels_20250826_213849.csv \
      --labels-out labels/labels_augmented.csv \
      --data-dir data/raw \
      --target-normal 350 \
      --fs-raw 25600 --fs-out 3200 --win-sec 1.0 --overlap 0.5 \
      --rpm 1770 --Z 9 --dmm 7.94 --Dmm 38.5 --theta 0 \
      --mains-hz 50 --mains-bw 10 --fmax 320 \
      --sev-thr 10 --strength-thr 0.05 --guard-sec 0.5
"""

import argparse, glob, os, time
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, hilbert, welch, decimate as sp_decimate

# ---------- безопасные импорты/фоллбэки из твоего проекта ----------
try:
    # если есть твой модуль с корректным чтением 3 фаз
    from features import sliding_windows as _sliding_windows, read_csv_3phase as _read3
except Exception:
    _sliding_windows, _read3 = None, None

# ---------- математика линий и огибающей ----------
def bearing_lines(rpm, Z, d_mm, D_mm, theta_deg):
    fr = rpm / 60.0
    theta = np.deg2rad(theta_deg)
    r = (d_mm / D_mm) * np.cos(theta)
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
    f, P = welch(env, fs=fs, nperseg=min(4096, len(env)))
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
        bwk = max(bw, 0.02 * fck)  # ≥ 2% от частоты
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

def choose_phase_for_env(w, fs, lines, mains_hz, bw, fmax):
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

# ---------- чтение и окна ----------
def read_csv_3phase(path):
    if _read3 is not None:
        return _read3(path)  # Nx3
    # универсальный фоллбэк
    df = pd.read_csv(path)
    cols = list(df.columns)
    # эвристики имён
    pref = {"A","B","C"}
    if set(cols[:3]).issuperset(pref):
        arr = df[list(sorted(pref))].to_numpy(dtype=np.float32)
    else:
        arr = df.iloc[:, :3].to_numpy(dtype=np.float32)
    return arr

def decimate_arr(x, fs_raw, fs_out):
    factor = max(1, int(round(fs_raw // fs_out)))
    if factor <= 1: return x, int(fs_raw)
    # пофазно
    y = np.zeros((int(np.ceil(x.shape[0]/factor)), x.shape[1]), dtype=np.float32)
    for i in range(x.shape[1]):
        y[:, i] = sp_decimate(x[:, i], factor, ftype='iir', zero_phase=True)
    return y, int(fs_raw//factor)

def sliding_windows(x, fs, win_sec, overlap):
    if _sliding_windows is not None:
        yield from _sliding_windows(x, fs, win_sec, overlap)
        return
    L = int(round(win_sec * fs))
    step = int(round(L * (1.0 - overlap)))
    for s in range(0, x.shape[0]-L+1, max(step,1)):
        yield s, x[s:s+L, :]

def window_id(path, i0, i1):
    return f"{os.path.basename(path)}:{i0}:{i1}"

# ---------- основная логика ----------
def collect_calm_normals_for_file(path, params, already_ids, defect_intervals, need_for_file):
    """
    Возвращает список записей-меток (только normal), не трогая существующие.
    """
    x_raw = read_csv_3phase(path)
    x, fs = decimate_arr(x_raw, params.fs_raw, params.fs_out)
    lines = bearing_lines(params.rpm, params.Z, params.dmm, params.Dmm, params.theta)

    L = int(params.win_sec * fs)
    out = []
    for s, w in sliding_windows(x, fs, params.win_sec, params.overlap):
        i0, i1 = s, s + L
        wid = window_id(path, i0, i1)
        if wid in already_ids:
            continue  # уже размечено

        # отступ от дефектных окон
        if defect_intervals:
            t0, t1 = i0/fs, i1/fs
            mid = 0.5*(t0+t1)
            too_close = any(abs(mid - 0.5*(a+b)) <= params.guard_sec for (a,b) in defect_intervals)
            if too_close:
                continue

        # выбор фазы и оценка «спокойности»
        idx = choose_phase_for_env(w, fs, lines, params.mains_hz, params.mains_bw, params.fmax)
        f, P = envelope_psd(w[:, idx], fs, fmax=params.fmax, mains_hz=params.mains_hz, bw=params.mains_bw)
        sev = severity_from_bands(f, P, lines)
        strength = defect_strength_ratio(f, P, lines)
        # строгие пороги для 100% норм
        if (sev <= params.sev_thr) and (strength <= params.strength_thr):
            rec = dict(
                win_id=wid, file=os.path.basename(path),
                i0=i0, i1=i1, fs=fs, t0=i0/fs, t1=i1/fs,
                y_defect=0, y_type="normal",
                severity=float(sev), confidence=0.70,
                notes=f"auto-normal-augment sev<={params.sev_thr},str<=${params.strength_thr}",
                rpm=float(params.rpm), Z=int(params.Z),
                dmm=float(params.dmm), Dmm=float(params.Dmm), theta=int(params.theta)
            )
            out.append(rec)

    # самые «спокойные» первыми — сортируем по (severity, strength)
    out.sort(key=lambda r: (r["severity"], r["confidence"]))  # conf одинаковая — фактически по sever.
    if need_for_file is not None:
        out = out[:max(0, need_for_file)]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--glob", default="*.csv")

    # целевые объёмы
    ap.add_argument("--target-normal", type=int, default=None,
                    help="Целевое общее число примеров класса normal после добавления. Если не задано — будет вычислено как медиана по дефектным классам, но не менее 300.")
    ap.add_argument("--n-per-file", type=int, default=None,
                    help="Жёсткая квота normal на файл (если задана, игнорирует target-normal).")

    # обработка сигналов
    ap.add_argument("--fs-raw", type=float, default=25600)
    ap.add_argument("--fs-out", type=float, default=3200)
    ap.add_argument("--win-sec", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.5)

    # демодуляция
    ap.add_argument("--mains-hz", type=float, default=50.0)
    ap.add_argument("--mains-bw", type=float, default=10.0)
    ap.add_argument("--fmax", type=float, default=320.0)

    # подшипник
    ap.add_argument("--rpm", type=float, default=1770)
    ap.add_argument("--Z", type=int, default=9)
    ap.add_argument("--dmm", type=float, default=7.94)
    ap.add_argument("--Dmm", type=float, default=38.5)
    ap.add_argument("--theta", type=int, default=0)

    # пороги и ограничения
    ap.add_argument("--sev-thr", type=float, default=10.0)
    ap.add_argument("--strength-thr", type=float, default=0.05)
    ap.add_argument("--guard-sec", type=float, default=0.5,
                    help="Отступ вокруг уже размеченных дефектных окон (сек), чтобы не брать спорные соседние окна.")

    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    labels_out = args.labels_out or f"labels/labels_augmented_{ts}.csv"
    os.makedirs(os.path.dirname(labels_out), exist_ok=True)

    df_lab = pd.read_csv(args.labels_in)
    df_lab["y_type"] = df_lab["y_type"].astype(str)

    # текущее состояние
    cur_norm = int((df_lab["y_type"] == "normal").sum())
    defect_counts = (df_lab[df_lab["y_type"] != "normal"]["y_type"]
                     .value_counts().to_dict())
    # целевой объём normal
    if args.n_per_file is None:
        if args.target_normal is not None:
            target_normal = int(args.target_normal)
        else:
            med = 0
            if len(defect_counts) > 0:
                med = int(np.median(list(defect_counts.values())))
            target_normal = max(300, med or 300)
        need_total = max(0, target_normal - cur_norm)
    else:
        need_total = None  # будем набирать n_per_file для каждого файла

    # подготовим индексы «не трогать»
    already_ids = set(df_lab["win_id"].astype(str).tolist())

    # интервалы дефектов по файлам (для guard-sec)
    defect_by_file = {}
    for _, r in df_lab.iterrows():
        if r["y_type"] != "normal":
            defect_by_file.setdefault(r["file"], []).append((float(r["t0"]), float(r["t1"])))

    # проход по файлам
    files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    added = []
    remain = need_total
    for p in files:
        fname = os.path.basename(p)
        defects_here = defect_by_file.get(fname, [])
        if args.n_per_file is not None:
            quota = max(0, int(args.n_per_file))
        else:
            if remain is None or remain <= 0:
                quota = 0
            else:
                # равномерное распределение остатка
                files_left = len(files) - files.index(p)
                quota = int(np.ceil(remain / max(1, files_left)))

        rows = collect_calm_normals_for_file(
            p, args, already_ids, defects_here, need_for_file=quota if quota>0 else None
        )
        added.extend(rows)
        if need_total is not None:
            remain -= len(rows)
            if remain <= 0:
                break

    # объединяем: существующие + новые (не перетираем)
    if len(added) == 0:
        print("Новых normal-окон не найдено по заданным порогам.")
        out = df_lab.copy()
    else:
        df_add = pd.DataFrame(added)
        out = pd.concat([df_lab, df_add], ignore_index=True)
        out.drop_duplicates(subset=["win_id"], keep="first", inplace=True)

    out.to_csv(labels_out, index=False)

    # сводка
    vc_before = df_lab["y_type"].value_counts().to_dict()
    vc_after  = out["y_type"].value_counts().to_dict()
    print(f"[OK] Сохранено: {labels_out}")
    print(f"  Было normal: {vc_before.get('normal',0)} → Стало normal: {vc_after.get('normal',0)} "
          f"(+{vc_after.get('normal',0)-vc_before.get('normal',0)})")
    print("  Распределение после добавления:", vc_after)

if __name__ == "__main__":
    main()
