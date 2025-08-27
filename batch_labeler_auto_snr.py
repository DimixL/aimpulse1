#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, time, datetime as dt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.signal import welch, butter, filtfilt, hilbert

# твои утилиты
from features import read_csv_3phase, decimate, sliding_windows, extract_features_window

LABELS  = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]

# ---------- частоты из геометрии ----------
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

# ---------- огибающая вокруг сети ----------
def envelope_psd(y, fs, fmax=320.0, mains_hz=60.0, bw=12.0):
    y = y - y.mean()
    lo_hz = max(1.0, mains_hz - bw)
    hi_hz = min(fs/2 - 1.0, mains_hz + bw)
    if hi_hz <= lo_hz:  # страховка
        lo_hz = max(0.5, hi_hz - 1.0)
    lo = lo_hz / (fs/2); hi = hi_hz / (fs/2)
    b, a = butter(4, [lo, hi], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, y)))
    f, P = welch(
        env, fs=fs,
        nperseg=min(4096, len(env)),
        detrend='constant',
        return_onesided=True
    )
    m = f <= fmax
    return f[m], P[m]

# ---------- гребёнка по SNR ----------
def comb_snr(f, P, fc, harmonics=3, w=None, search_rel=0.08, w_min=1.0):
    if fc <= 0 or fc >= f[-1]*0.98:
        return 0.0, 0.0
    snrs, e_sig_sum = [], 0.0
    for k in range(1, harmonics+1):
        fck = fc * k
        if fck >= f[-1]: break
        wk = max(w_min, (w if w is not None else 0.05 * fck))  # было 0.02 → 0.05 и минимум
        best_snr, best_e = 0.0, 0.0
        # локальный поиск центра гребня вокруг прогнозной частоты
        for fc_try in np.linspace(fck*(1-search_rel), fck*(1+search_rel), 7):
            m_sig = (f >= fc_try - wk) & (f <= fc_try + wk)
            if not np.any(m_sig):
                continue
            e_sig = float(np.trapezoid(P[m_sig], f[m_sig]))
            # шум — дальше от полосы
            m_noise = ((f >= fc_try - 6*wk) & (f <= fc_try - 3*wk)) | ((f >= fc_try + 3*wk) & (f <= fc_try + 6*wk))
            if not np.any(m_noise):
                continue
            e_noise_d = float(np.trapezoid(P[m_noise], f[m_noise])) / (np.sum(m_noise) + 1e-9)
            e_sig_d   = e_sig / (np.sum(m_sig) + 1e-9)
            snr = e_sig_d / (e_noise_d + 1e-12)
            if snr > best_snr:
                best_snr, best_e = snr, e_sig
        if best_snr > 0:
            snrs.append(best_snr); e_sig_sum += best_e
    if not snrs:
        return 0.0, 0.0
    return e_sig_sum, float(np.median(snrs))  # median устойчивее mean

def severity_from_defect_bands(f,P,lines):
    e = 0.0
    for k in ["BPFO","BPFI","BSF","FTF"]:
        ei, _ = comb_snr(f,P,lines[k], harmonics=3)
        e += ei
    denom = _area(f,P) + 1e-12
    return float(np.clip(400.0 * e/denom, 0, 100))

# ---------- выбор фазы ----------
def choose_phase_for_env(w, fs, lines_hint, mains_hz, bw, fmax):
    best_idx, best_val = 0, -1.0
    for idx in range(min(w.shape[1], 3)):
        f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mains_hz, bw=bw)
        val = 0.0
        for k in ["BPFO","BPFI","BSF","FTF"]:
            _, snr = comb_snr(f,P,lines_hint[k], harmonics=2)
            val += snr
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx

# ---------- оценка RPM по огибающей (1×) ----------
def estimate_fr_from_env(f, P, fr_hint, rel_tol=0.2):
    lo, hi = fr_hint*(1-rel_tol), fr_hint*(1+rel_tol)
    m = (f >= lo) & (f <= hi)
    if not np.any(m): return None
    # игнорируем совсем низкие частоты < 1 Гц
    fm = f[m]; Pm = P[m].copy()
    Pm[fm < 1.0] = 0.0
    if Pm.max() <= 0: return None
    return float(fm[np.argmax(Pm)])

# ---------- автоопределение сети ----------
def detect_mains_psd(x, fs, bw=1.0):
    f, Psum = None, None
    for k in range(min(x.shape[1], 3)):
        fk, Pk = welch(
            x[:, k] - x[:, k].mean(),
            fs=fs,
            nperseg=min(8192, len(x)),
            detrend='constant',
            return_onesided=True
        )
        f = fk if f is None else f
        Psum = Pk if Psum is None else Psum + Pk
    def en(fc):
        m = (f >= fc-bw) & (f <= fc+bw)
        return float(Psum[m].sum())
    e50, e60 = en(50.0), en(60.0)
    return (50.0 if e50>=e60 else 60.0), e50, e60

def detect_mains_by_defect_energy(w, fs, lines, fmax, mains_bw=12.0):
    best_hz, best_s = 50.0, -1
    for mhz in (50.0, 60.0):
        # грубая фаза A достаточно для сравнения
        f, P = envelope_psd(w[:,0], fs, fmax=fmax, mains_hz=mhz, bw=mains_bw)
        s = 0.0
        for k in ["BPFO","BPFI","BSF","FTF"]:
            _, snr = comb_snr(f,P,lines[k],harmonics=3)
            s += snr
        if s > best_s: best_s, best_hz = s, mhz
    return best_hz

import re
def _extract_from_notes(notes, key, default=np.nan):
    # key: 'snr' или 'e_rel'
    try:
        m = re.search(rf"{key}=([0-9.]+)", notes or "")
        return float(m.group(1)) if m else default
    except Exception:
        return default

def post_filter_types(rows, args):
    # сортируем по SNR убыв.
    rows_sorted = sorted(rows, key=lambda r: _extract_from_notes(r.get("notes",""), "snr", 0.0), reverse=True)

    kept = []
    last_span_by_key = {}   # (file, y_type) -> список оставленных интервалов
    count_by_key = {}       # (file, y_type) -> сколько уже оставили

    for r in rows_sorted:
        lab  = r["y_type"]
        fr   = float(r.get("rpm_est", r["rpm"])) / 60.0
        mains = float(r.get("mains_hz", 50.0))
        t0, t1 = float(r["t0"]), float(r["t1"])
        snr   = _extract_from_notes(r.get("notes",""), "snr", 0.0)
        e_rel = _extract_from_notes(r.get("notes",""), "e_rel", np.nan)
        sev   = float(r.get("severity", 0.0))

        # --- близость к сети: режем наводки 1×/2×RPM
        # misalignment ~ 2×fr; imbalance ~ 1×fr
        if lab == "misalignment":
            if abs(2.0*fr - mains) < args.misal_mains_gap:
                continue
        elif lab == "imbalance":
            if abs(fr - mains) < args.imb_mains_gap:
                continue

        # --- подшипники: требуем минимальные признаки
        if lab in ("BPFO", "BPFI", "BSF", "FTF"):
            ok = (sev >= args.sev_min_bearing) and np.isfinite(e_rel) and (e_rel >= args.min_e_rel_bearing)
            if not ok:
                # резерв: если гребёнка очень уверенная, пропускаем e_rel/severity
                if snr < args.backup_snr_bearing:
                    continue

        # --- NMS: не даём окнам одного типа по одному файлу перекрываться слишком плотно
        key = (r["file"], lab)
        lst = last_span_by_key.setdefault(key, [])
        overlap = False
        for (u0,u1) in lst:
            if not (t1 <= u0 - args.min_gap_sec or t0 >= u1 + args.min_gap_sec):
                overlap = True
                break
        if overlap:
            continue

        # --- лимит количества на файл и тип
        cnt = count_by_key.get(key, 0)
        if cnt >= args.per_file_maxk:
            continue

        lst.append((t0,t1))
        count_by_key[key] = cnt + 1
        kept.append(r)

    return kept

# ---------- обработка одного файла ----------
def process_one_file(path, args):
    # читаем исходник (Nx3)
    x = read_csv_3phase(path)

    # согласование дискретизации
    ratio  = float(args.fs_raw) / float(args.fs_out)
    factor = max(1, int(round(ratio)))
    x2     = decimate(x, factor) if factor > 1 else x
    fs     = int(round(float(args.fs_raw) / factor))

    # номинальные линии по подсказке RPM (для выбора фазы/сети)
    lines_hint = bearing_lines(args.rpm, args.Z, args.dmm, args.Dmm, args.theta)

    # определяем 50/60 Гц
    mains_hz, e50, e60 = detect_mains_psd(x2, fs, bw=1.0)
    if abs(e50 - e60) / (max(e50, e60) + 1e-12) < 0.15:
        L  = int(args.win * fs)
        s0 = 0 if len(x2) < L else (len(x2)//2 - L//2)
        mains_hz = detect_mains_by_defect_energy(x2[s0:s0+L], fs, lines_hint, args.fmax, args.mains_bw)

    # окна
    spans = [(s, s + int(args.win*fs)) for s, _ in sliding_windows(x2, fs, args.win, args.overlap)]
    spans_kept = []
    feats = []
    rpm_est_list = []
    carrier_list = []
    phase_idx_list = []
    severities, strengths = [], []
    type_energy_list, type_snr_list = [], []

    # основной цикл по окнам
    for s, w in sliding_windows(x2, fs, args.win, args.overlap):

        # (1) грубый выбор фазы по номинальным линиям
        idx = choose_phase_for_env(w, fs, lines_hint, mains_hz, args.mains_bw, args.fmax)

        # (2) демодуляция: тестируем 1× и 2× сети, выбираем лучшую по сумме дефектной энергии
        best = None
        for carrier in (mains_hz, 2.0 * mains_hz):
            f_try, P_try = envelope_psd(w[:, idx], fs, fmax=args.fmax, mains_hz=carrier, bw=args.mains_bw)
            if f_try.size == 0:  # нет спектра огибающей — пропускаем
                continue
            total_try = _area(f_try, P_try) + 1e-12
            e_sum = 0.0
            for fc, h in [(lines_hint["BPFO"], 3), (lines_hint["BPFI"], 3), (lines_hint["BSF"], 3),
                          (lines_hint["FTF"], 3)]:
                e_i, _ = comb_snr(f_try, P_try, fc, harmonics=h)
                e_sum += e_i
            score = e_sum / total_try
            if (best is None) or (score > best[0]):
                best = (score, f_try, P_try, carrier, idx)

        if best is None:
            # ничего в огибающей — окно пропускаем
            continue

        _, f, P, carrier_chosen, idx = best

        feats.append(extract_features_window(w, fs))
        spans_kept.append((s, s + int(args.win * fs)))
        # (3) оценка реального RPM в этом окне на выбранной огибающей
        fr_hint = args.rpm / 60.0
        fr_est  = estimate_fr_from_env(f, P, fr_hint, rel_tol=args.rpm_rel_tol)
        rpm_use = (fr_est * 60.0) if (fr_est is not None) else args.rpm
        lines   = bearing_lines(rpm_use, args.Z, args.dmm, args.Dmm, args.theta)
        rpm_est_list.append(float(rpm_use))
        carrier_list.append(float(carrier_chosen))
        phase_idx_list.append(int(idx))

        # (4) гребёнки по типам
        total = _area(f, P) + 1e-12
        sc_e, sc_snr = {}, {}
        for lab, (fc, h) in {
            "BPFO":        (lines["BPFO"], 3),
            "BPFI":        (lines["BPFI"], 3),
            "BSF":         (lines["BSF"],  3),
            "FTF":         (lines["FTF"],  3),
            "imbalance":    (lines["fr"],     2),
            "misalignment": (2.0*lines["fr"], 2),
        }.items():
            e_sum, snr = comb_snr(f, P, fc, harmonics=h)
            sc_e[lab]   = e_sum / total
            sc_snr[lab] = snr

        # (5)severity и «сила подшипников»
        sev  = severity_from_defect_bands(f, P, lines)
        strg = sc_e["BPFO"] + sc_e["BPFI"] + sc_e["BSF"] + sc_e["FTF"]

        severities.append(sev)
        strengths.append(strg)
        type_energy_list.append(sc_e)
        type_snr_list.append(sc_snr)

    # матрица фич/метрик
    X = np.array(feats, np.float32)
    severities = np.array(severities, np.float32)
    strengths  = np.array(strengths,  np.float32)

    # авто-нормы
    normals_idx = []
    if len(X) >= 5:
        oc = IsolationForest(n_estimators=300, contamination=args.contam, random_state=42).fit(X)
        anom = -oc.decision_function(X)
    else:
        anom = np.zeros(len(X), dtype=float)

    calm = (severities <= args.sev_thr) & (strengths <= args.strength_thr)
    pool = np.where(calm)[0]
    if pool.size:
        order = pool[np.argsort(anom[pool])]
        normals_idx = list(order[:min(args.n_norm, len(order))])

    # кандидаты по типам (динамич. пороги)
    rows_types = []
    labels_hunt = ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]

    snr_by_label = {lab: [] for lab in labels_hunt}
    for j, sc in enumerate(type_snr_list):
        for lab in labels_hunt:
            snr_by_label[lab].append((sc[lab], j))

    for lab in labels_hunt:
        arr = snr_by_label[lab]
        if not arr:
            continue
        vals = np.array([v for v, _ in arr], float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            thr = (args.min_snr_bearing if lab in ("BPFO","BPFI","BSF","FTF")
                   else (args.min_snr_misalignment if lab=="misalignment" else 1.25))
        else:
            if lab in ("BPFO","BPFI","BSF","FTF"):
                thr = max(args.min_snr_bearing, float(np.nanpercentile(vals, args.perc_q_bearing)))
            elif lab == "misalignment":
                thr = max(args.min_snr_misalignment, float(np.nanpercentile(vals, args.perc_q)))
            else:
                thr = max(1.25, float(np.nanpercentile(vals, 85.0)))

        cand = [(v,j) for v,j in arr if np.isfinite(v) and v >= thr]
        if not cand:
            cand = sorted(arr, key=lambda x: -x[0])[:max(3, args.topk//2)]
        else:
            cand.sort(key=lambda x: -x[0])
            cand = cand[:args.topk]

        for snr, j in cand:
            s0, s1 = spans_kept[j]
            e_rel = type_energy_list[j][lab]

            # вычисляем confidence вне словаря (логистическая шкала)
            conf = 1.0 / (1.0 + np.exp(-0.35 * (snr - 1.0)))
            conf = float(np.clip(conf, 0.30, 0.95))
            if not np.isfinite(conf):
                conf = 0.30

            rows_types.append(dict(
                win_id=f"{os.path.basename(path)}:{s0}:{s1}",
                file=os.path.basename(path), i0=s0, i1=s1, fs=fs,
                t0=s0 / fs, t1=s1 / fs,
                y_defect=int(lab != "normal"), y_type=lab,
                severity=float(severities[j]),
                confidence=conf,
                notes=f"auto-{lab} snr={snr:.2f} e_rel={e_rel:.4f}",
                rpm=float(args.rpm), Z=int(args.Z), dmm=float(args.dmm),
                Dmm=float(args.Dmm), theta=int(args.theta),
                mains_hz=float(mains_hz),
                rpm_est=float(rpm_est_list[j]),
                carrier_hz=float(carrier_list[j]),
                phase_idx=int(phase_idx_list[j]),
            ))

    # нормальные окна
    rows_norm = []
    for j in normals_idx:
        s0, s1 = spans_kept[j]
        rows_norm.append(dict(
            win_id=f"{os.path.basename(path)}:{s0}:{s1}",
            file=os.path.basename(path), i0=s0, i1=s1, fs=fs,
            t0=s0/fs, t1=s1/fs, y_defect=0, y_type="normal",
            severity=float(severities[j]), confidence=0.5,
            notes="auto-normal",
            rpm=float(args.rpm), Z=int(args.Z), dmm=float(args.dmm),
            Dmm=float(args.Dmm), theta=int(args.theta),
            mains_hz=float(mains_hz),
            rpm_est=float(rpm_est_list[j]) if j < len(rpm_est_list) else float(args.rpm),
        ))
    rows_types = post_filter_types(rows_types, args)
    return rows_norm + rows_types, mains_hz


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Batch labeling with per-window RPM estimation + SNR comb")
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out-dir",   required=True)
    # input & bearing
    p.add_argument("--fs-raw", type=float, default=25600.0)
    p.add_argument("--rpm", type=float, default=1770.0)
    p.add_argument("--Z",   type=int,   default=9)
    p.add_argument("--dmm", type=float, default=7.94)
    p.add_argument("--Dmm", type=float, default=38.5)
    p.add_argument("--theta", type=float, default=0.0)
    # signal/demod
    p.add_argument("--fs-out",   type=int,   default=5120)
    p.add_argument("--win",      type=float, default=2.0)
    p.add_argument("--overlap",  type=float, default=0.5)
    p.add_argument("--fmax",     type=float, default=320.0)
    p.add_argument("--mains-bw", type=float, default=20.0)
    p.add_argument("--rpm-rel-tol", type=float, default=0.30, help="допуск поиска 1×RPM (доля от подсказки)")
    # auto-normal
    p.add_argument("--n-norm",        type=int,   default=15)
    p.add_argument("--sev-thr",       type=float, default=10.0)
    p.add_argument("--strength-thr",  type=float, default=0.05)
    p.add_argument("--contam",        type=float, default=0.05)
    # candidates per type
    p.add_argument("--topk",          type=int,   default=40)
    p.add_argument("--perc-q",        type=float, default=90.0)  # для misalignment
    p.add_argument("--perc-q-bearing",type=float, default=80.0)  # для BPFO/BPFI/BSF/FTF
    p.add_argument("--min-snr-bearing",      type=float, default=1.10)
    p.add_argument("--min-snr-misalignment", type=float, default=1.60)
    p.add_argument("--sev-min-bearing", type=float, default=6.0, help="минимальная severity для BPFO/BPFI/BSF/FTF")
    p.add_argument("--min-e-rel-bearing", type=float, default=0.004,
                   help="минимальная относительная энергия дефектной полосы")
    p.add_argument("--imb-mains-gap", type=float, default=3.0, help="мин. разнос между fr и сетью, Гц (imbalance)")
    p.add_argument("--misal-mains-gap", type=float, default=3.0,
                   help="мин. разнос между 2*fr и сетью, Гц (misalignment)")
    p.add_argument("--min-gap-sec", type=float, default=1.0,
                   help="NMS: мин. разнос окон одного типа по одному файлу, с")
    p.add_argument("--per-file-maxk", type=int, default=50, help="макс. окон одного типа на файл после фильтра")
    p.add_argument("--backup-snr-bearing", type=float, default=1.40,
                   help="если e_rel/severity не прошли, но SNR ≥ этого порога — оставить окно (спасает слабые BPFO/BPFI/BSF)")
    return p.parse_args()

def main():
    args = parse_args()
    files = sorted(glob.glob(args.data_glob))
    assert files, f"Нет файлов по маске: {args.data_glob}"

    t0 = time.time()
    all_rows, mains_stat = [], {50.0:0, 60.0:0}

    for p in files:
        rows, mains_hz = process_one_file(p, args)
        all_rows.extend(rows)
        mains_stat[50.0 if abs(mains_hz-50.0) < 0.6 else 60.0] += 1
        print(f"[{os.path.basename(p)}] mains≈{mains_hz:.1f} Hz -> labels {len(rows)}")

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

    # короткий топ по SNR для отладки (по всему датасету)
    if not df.empty:
        print("Top-3 окна по каждому типу (для контроля):")
        for lab in ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]:
            sub = df[df["y_type"]==lab].copy()
            if "notes" in sub.columns:
                # выдёргиваем snr из notes
                try:
                    sub["snr"] = sub["notes"].str.extract(r"snr=([0-9.]+)").astype(float)
                except Exception:
                    sub["snr"] = np.nan
            sub = sub.sort_values("snr", ascending=False).head(3)
            if len(sub)==0:
                print(f"  {lab:<12}: —")
            else:
                for _,r in sub.iterrows():
                    print(f"  {lab:<12}: {r['file']}  t=[{r['t0']:.2f},{r['t1']:.2f}]  snr={r.get('snr',np.nan):.2f}  sev={r['severity']:.1f}")

if __name__ == "__main__":
    main()
