# labeler_app.py
import streamlit as st
import pandas as pd
import numpy as np
import glob, os, io, json
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from scipy.signal import iirnotch, filtfilt, hilbert, welch, butter
import plotly.graph_objects as go

st.set_page_config(page_title="ИИмпульс • Разметка CSV", layout="wide")

# ---- параметры по умолчанию ----
FS_RAW = 25600           # как в твоих страницах
FS_OUT = 3200            # частота обработки
WIN_SEC = 1.0
OVERLAP = 0.5
TOP_N = 30               # сколько кандидатов показывать
LABELS = ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]

#  RU ↔ EN метки и хелперы для UI
LABELS_EN = LABELS  # совместимость, используем тот же список
LABELS_RU = {
    "normal": "норма",
    "BPFO": "дефект наружного кольца (BPFO)",
    "BPFI": "дефект внутреннего кольца (BPFI)",
    "BSF": "дефект тел качения (BSF)",
    "FTF": "дефект сепаратора (FTF)",
    "imbalance": "дисбаланс (1×RPM)",
    "misalignment": "расцентровка (2×RPM)",
}

def _area(f, P, m=None):
    # интеграл по всему спектру или по маске m
    if m is None:
        return float(np.trapezoid(P, f))
    return float(np.trapezoid(P[m], f[m]))


def to_ru(label_en: str) -> str:
    return LABELS_RU.get(label_en, label_en)

def to_en(label_ru: str) -> str:
    rev = {v: k for k, v in LABELS_RU.items()}
    return rev.get(label_ru, label_ru)

# ---- твои утилиты ----
from features import read_csv_3phase, decimate, sliding_windows, extract_features_window

# ---- helpers ----
def bearing_lines(rpm, Z, d_mm, D_mm, theta_deg):
    fr = rpm / 60.0
    theta = np.deg2rad(theta_deg)
    r = (d_mm / D_mm) * np.cos(theta)
    FTF  = 0.5 * fr * (1 - r)
    BPFO = 0.5 * Z * fr * (1 - r)
    BPFI = 0.5 * Z * fr * (1 + r)
    BSF  = (D_mm / d_mm) * fr * 0.5 * (1 - r**2)
    return dict(fr=fr, FTF=FTF, BPFO=BPFO, BPFI=BPFI, BSF=BSF)

def envelope_psd(y, fs, fmax=320.0, mains_hz=60.0, bw=10.0):
    y = y - y.mean()
    lo = max(1.0, mains_hz - bw) / (fs/2)
    hi = min(fs/2 - 1.0, mains_hz + bw) / (fs/2)
    b, a = butter(4, [lo, hi], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, y)))
    f, P = welch(env, fs=fs, nperseg=min(4096, len(env)))
    m = f <= fmax
    return f[m], P[m]


def band_energy(f, P, fc, bw=2.0, harmonics=1):
    """
    Интегрируем энергию в окрестности частоты fc и её гармоник.
    Ширина окна масштабируется от частоты.
    """
    if fc <= 0:
        return 0.0
    e = 0.0
    for k in range(1, harmonics+1):
        fck = fc * k
        if fck <= 0:
            continue
        # ± max(1 Гц, 2% от частоты)
        bwk = max(bw, 0.02 * fck)
        m = (f >= fck - bwk) & (f <= fck + bwk)
        if m.any():
            e += _area(f, P, m)

    return e


def autosuggest_label(f, P, lines):
    # доминирующая дефектная полоса среди BPFO/BPFI/BSF/FTF и «механика»: 1×/2×RPM
    e = {}
    e["BPFO"] = band_energy(f,P,lines["BPFO"])
    e["BPFI"] = band_energy(f,P,lines["BPFI"])
    e["BSF"]  = band_energy(f,P,lines["BSF"])
    e["FTF"]  = band_energy(f,P,lines["FTF"])
    e["1x"]   = band_energy(f,P,lines["fr"])
    e["2x"]   = band_energy(f,P,2*lines["fr"])
    # эвристики
    major = max(e, key=e.get)
    if major == "1x": return "imbalance", e
    if major == "2x": return "misalignment", e
    if major in ["BPFO","BPFI","BSF","FTF"]: return major, e
    return "normal", e

def severity_from_bands(f,P,lines):
    bands = ["BPFO","BPFI","BSF","FTF"]
    numer = sum(band_energy(f,P,lines[b]) for b in bands)
    denom = _area(f, P) + 1e-12
    return float(np.clip(400.0 * numer/denom, 0, 100))  # та же нормировка, что у тебя

def recommend_by_rules(f, P, lines):
    """Возвращает: (y_type, severity, confidence, flag, why_df)"""
    bands = {
        "BPFO": band_energy(f, P, lines["BPFO"]),
        "BPFI": band_energy(f, P, lines["BPFI"]),
        "BSF":  band_energy(f, P, lines["BSF"]),
        "FTF":  band_energy(f, P, lines["FTF"]),
        "1x":   band_energy(f, P, lines["fr"]),
        "2x":   band_energy(f, P, 2*lines["fr"]),
    }
    # выбор класса
    major = max(bands, key=bands.get)
    if major == "1x":
        y_type = "imbalance"
    elif major == "2x":
        y_type = "misalignment"
    else:
        y_type = major if major in ["BPFO", "BPFI", "BSF", "FTF"] else "normal"

    # severity — доля энергии «дефектных» полос
    sev = severity_from_bands(f, P, lines)

    # уверенность = доминирование топ-полосы + сила дефектных полос
    sorted_vals = sorted(bands.values(), reverse=True)
    dom = sorted_vals[0] / (sorted_vals[1] + 1e-9)  # лидер/второй
    def_en = bands["BPFO"] + bands["BPFI"] + bands["BSF"] + bands["FTF"]
    total = _area(f, P) + 1e-12
    strength = def_en / total
    conf = float(np.clip(0.4 * dom + 0.6 * np.clip(strength / 0.15, 0, 1), 0, 1))

    flag = "RED" if sev >= 80 else ("ORANGE" if sev >= 60 else "GREEN")

    why_df = pd.DataFrame(
        {"Полоса": ["BPFO", "BPFI", "BSF", "FTF", "1×RPM", "2×RPM"],
         "Энергия": [bands["BPFO"], bands["BPFI"], bands["BSF"], bands["FTF"], bands["1x"], bands["2x"]]}
    ).sort_values("Энергия", ascending=False)

    return y_type, float(sev), conf, flag, why_df


def window_id(path, i0, i1):
    return f"{os.path.basename(path)}:{i0}:{i1}"

@st.cache_data(show_spinner=False)
def list_files(pattern="data/raw/*.csv"):
    return sorted(glob.glob(pattern))

@st.cache_data(show_spinner=True)
def load_and_resample(path, fs_out=FS_OUT):
    x = read_csv_3phase(path)   # Nx3
    factor = max(1, FS_RAW // fs_out)
    x2 = decimate(x, factor) if factor>1 else x
    fs = FS_RAW // factor
    return x2.astype(np.float32), fs

@st.cache_data(show_spinner=True)
def make_windows(x, fs, win_sec=WIN_SEC, overlap=OVERLAP):
    # возвращаем [(start_idx, end_idx)]
    idxs = [ (s, s+int(win_sec*fs)) for s,_ in sliding_windows(x, fs, win_sec, overlap) ]
    return idxs

@st.cache_data(show_spinner=True)
def rank_anomalies(x, fs, win_sec=WIN_SEC, overlap=OVERLAP, top_n=TOP_N, seed=42):
    feats = []
    spans = []
    for s, w in sliding_windows(x, fs, win_sec, overlap):
        feats.append(extract_features_window(w, fs))
        spans.append((s, s+int(win_sec*fs)))
    X = np.array(feats, dtype=np.float32)
    if len(X) < 5:
        scores = np.zeros(len(X))
    else:
        oc = IsolationForest(n_estimators=300, contamination=0.05, random_state=seed)
        oc.fit(X)
        scores = -oc.decision_function(X)  # больше → аномальнее
    order = np.argsort(-scores)[:min(top_n, len(scores))]
    return order, spans, scores

def load_labels(path="labels/labels.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "win_id","file","i0","i1","t0","t1","fs",
        "y_defect","y_type","severity","confidence","notes",
        "rpm","Z","dmm","Dmm","theta"
    ])

def upsert_label(df_labels, rec, path="labels/labels.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rec["win_id"] in set(df_labels["win_id"]):
        df_labels.loc[df_labels["win_id"]==rec["win_id"], list(rec.keys())] = list(rec.values())
    else:
        df_labels = pd.concat([df_labels, pd.DataFrame([rec])], ignore_index=True)
    df_labels.to_csv(path, index=False)
    return df_labels

# ==== [PATCH 1] Helpers: выбор фазы, подсчёт скорингов, авто-норма и кандидаты по типам ====
def choose_phase_for_env(w: np.ndarray, fs: int, lines: dict, mains_hz: float, bw: float, fmax: float) -> int:
    best_idx, best_val = 0, -1.0
    for idx in range(min(w.shape[1], 3)):
        f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mains_hz, bw=bw)
        val = sum(band_energy(f, P, lines[k]) for k in ["BPFO","BPFI","BSF","FTF"])
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx

def compute_window_scores(y: np.ndarray, fs: int, lines: dict, mains_hz: float, bw: float, fmax: float):
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

@st.cache_data(show_spinner=True)
def scan_file_features(x: np.ndarray, fs: int, win_sec: float, overlap: float,
                       rpm: float, Z: int, dmm: float, Dmm: float, theta: float,
                       mains_hz: float, mains_bw: float, fmax: float):
    lines = bearing_lines(rpm, Z, dmm, Dmm, theta)
    spans, feats, severities, strengths, type_scores = [], [], [], [], []
    L = int(win_sec * fs)
    for s, w in sliding_windows(x, fs, win_sec, overlap):
        spans.append((s, s + L))
        feats.append(extract_features_window(w, fs))
        idx = choose_phase_for_env(w, fs, lines, mains_hz, mains_bw, fmax)
        sev, sc, strg = compute_window_scores(w[:, idx], fs, lines, mains_hz, mains_bw, fmax)
        severities.append(sev); strengths.append(strg); type_scores.append(sc)
    return spans, np.array(feats, np.float32), np.array(severities), np.array(strengths), type_scores


def autolabel_normals_for_file(path: str, *, fs_out=FS_OUT, win_sec=WIN_SEC, overlap=OVERLAP,
                               n_per_file=15, sev_thr=10.0, strength_thr=0.05, contam=0.05,
                               rpm=None, Z=None, dmm=None, Dmm=None, theta=None, mains_hz=None, mains_bw=None, fmax=None):
    x, fs = load_and_resample(path, fs_out)
    # подставим дефолты из «глобальных» UI, если не передали явно
    fmax_use = fmax if fmax is not None else 320.0
    rp = rpm if rpm is not None else globals()["rpm"]
    z  = Z if Z is not None else globals()["Z"]
    dm = dmm if dmm is not None else globals()["dmm"]
    Dm = Dmm if Dmm is not None else globals()["Dmm"]
    th = theta if theta is not None else globals()["theta"]
    mh = mains_hz if mains_hz is not None else globals()["mains_hz"]
    mb = mains_bw if mains_bw is not None else globals()["mains_bw"]

    spans, X, sevs, strengths, _ = scan_file_features(
        x, fs, win_sec, overlap, rp, z, dm, Dm, th, mh, mb, fmax_use
    )
    # аномальность по IsolationForest
    if len(X) >= 5:
        oc = IsolationForest(n_estimators=300, contamination=contam, random_state=42).fit(X)
        anom = -oc.decision_function(X)  # больше → аномальнее
    else:
        anom = np.zeros(len(X), dtype=float)

    # фильтр «спокойных»
    mask = (sevs <= sev_thr) & (strengths <= strength_thr)
    if not np.any(mask):
        return 0

    idx_pool = np.where(mask)[0]
    order = idx_pool[np.argsort(anom[idx_pool])]  # самые спокойные
    picked = order[:min(n_per_file, len(order))]

    df_lab = load_labels()
    saved = 0
    for k in picked:
        i0, i1 = spans[k]
        rec = {
            "win_id": window_id(path, i0, i1),
            "file": os.path.basename(path),
            "i0": i0, "i1": i1, "fs": fs,
            "t0": i0 / fs, "t1": i1 / fs,
            "y_defect": 0,
            "y_type": "normal",
            "severity": float(sevs[k]),     # уже 0..100
            "confidence": 0.5,
            "notes": "auto-normal",
            "rpm": float(rp), "Z": int(z), "dmm": float(dm), "Dmm": float(Dm), "theta": int(th)
        }
        df_lab = upsert_label(df_lab, rec)
        saved += 1


    return saved

def autolabel_types_for_file(path: str, *, fs_out=FS_OUT, win_sec=WIN_SEC, overlap=OVERLAP,
                             labels=("BPFO","BPFI","BSF","FTF","imbalance","misalignment"),
                             top_k=10, min_score=0.02,
                             rpm=None, Z=None, dmm=None, Dmm=None, theta=None, mains_hz=None, mains_bw=None, fmax=None):
    x, fs = load_and_resample(path, fs_out)
    rp = rpm if rpm is not None else globals()["rpm"]
    z  = Z if Z is not None else globals()["Z"]
    dm = dmm if dmm is not None else globals()["dmm"]
    Dm = Dmm if Dmm is not None else globals()["Dmm"]
    th = theta if theta is not None else globals()["theta"]
    mh = mains_hz if mains_hz is not None else globals()["mains_hz"]
    mb = mains_bw if mains_bw is not None else globals()["mains_bw"]
    fmax_use = fmax if fmax is not None else 320.0

    spans, _, sevs, _, type_scores = scan_file_features(
        x, fs, win_sec, overlap, rp, z, dm, Dm, th, mh, mb, fmax_use
    )

    buckets = {lab: [] for lab in labels}
    for idx, sc in enumerate(type_scores):
        for lab in labels:
            buckets[lab].append((sc[lab], idx))  # (score, window_idx)

    df_lab = load_labels()
    saved = 0
    for lab, arr in buckets.items():
        arr = [a for a in arr if a[0] >= min_score]
        arr.sort(key=lambda x: -x[0])
        for score, j in arr[:top_k]:
            i0, i1 = spans[j]
            rec = {
                "win_id": window_id(path, i0, i1),
                "file": os.path.basename(path),
                "i0": i0, "i1": i1, "fs": fs,
                "t0": i0 / fs, "t1": i1 / fs,
                "y_defect": int(lab != "normal"),
                "y_type": lab,
                "severity": float(sevs[j]),   # 0..100
                "confidence": float(np.clip(score / max(min_score, 1e-6), 0.3, 1.0)),  # грубая калибровка
                "notes": f"auto-{lab} score={score:.3f}",
                "rpm": float(rp), "Z": int(z), "dmm": float(dm), "Dmm": float(Dm), "theta": int(th)
            }
            df_lab = upsert_label(df_lab, rec)
            saved += 1

    return saved
# ==== [END PATCH 1] ====

# ------------------- UI -------------------
st.title("🧩 ИИмпульс • Разметка CSV")
st.caption("Скрининг аномалий → weak-labels правилами → ручная правка. Сохранение в labels/labels.csv")

files = list_files()
if not files:
    st.warning("Положите данные в data/raw/*.csv")
    st.stop()

c0, c1, c2, c3 = st.columns([2,1,1,1])
with c0:
    fsel = st.selectbox("Файл (двигатель)", files, format_func=os.path.basename)
with c1:
    fs_out = st.select_slider("Fs, Гц", [2000,3200,5120], value=FS_OUT)
with c2:
    win_sec = st.select_slider("Окно, с", [0.5,1.0,2.0], value=WIN_SEC)
with c3:
    top_n = st.select_slider("Top-аномалий", [10,20,30,40,50], value=TOP_N)

with st.expander("Параметры подшипника / RPM", expanded=False):
    rpm   = st.number_input("RPM", value=1770, step=10)
    Z     = st.number_input("Число тел качения Z", value=9, step=1)
    dmm   = st.number_input("Диаметр шарика d, мм", value=7.94, step=0.01)
    Dmm   = st.number_input("Диаметр делит. окружности D, мм", value=38.5, step=0.1)
    theta = st.slider("Угол контакта, °", 0, 30, 0)

with st.expander("Демодуляция (сеть)", expanded=False):
    mains_hz = st.slider("Частота сети, Гц", 45.0, 65.0, 60.0, 0.5)   # дефолт 60 Гц
    mains_bw = st.slider("Полоса вокруг сети, Гц", 2.0, 20.0, 10.0, 0.5)
    fmax_env = st.slider("Макс. частота огибающей, Гц (текущее окно)", 200.0, 400.0, 320.0, 10.0)

lines = bearing_lines(rpm,Z,dmm,Dmm,theta)

x, fs = load_and_resample(fsel, fs_out)
order, spans, scores = rank_anomalies(x, fs, win_sec=win_sec, overlap=OVERLAP, top_n=top_n)

st.markdown("### Кандидаты")
cand = pd.DataFrame({
    "#": np.arange(len(order)),
    "t0": [spans[i][0]/fs for i in order],
    "t1": [spans[i][1]/fs for i in order],
    "score": [float(scores[i]) for i in order],
})
st.dataframe(cand, use_container_width=True, height=220)

sel = st.number_input("Выбери индекс кандидата", min_value=0, max_value=max(0,len(order)-1), value=0, step=1)

if len(order)==0:
    st.info("Аномалии не найдены для текущих настроек.")
    st.stop()

i = order[int(sel)]
i0, i1 = spans[i]
w = x[i0:i1]                 # окно Nx3
t = np.arange(i0, i1)/fs
win_dur = (i1-i0)/fs

# --- график времени (3 фазы) ---
fig_ts = go.Figure()
for k, name in enumerate(["A","B","C"]):
    fig_ts.add_trace(go.Scattergl(x=t, y=w[:,k], mode="lines", name=name, line=dict(width=1)))
fig_ts.update_layout(height=240, xaxis_title="Время, с", yaxis_title="Ток, A", legend=dict(orientation="h"))
st.plotly_chart(fig_ts, use_container_width=True)

# --- огибающая PSD по выбранной фазе ---
ph = st.radio("Фаза для огибающей", ["Auto","A","B","C"], horizontal=True, index=0, key="env_phase")
if ph == "Auto":
    idx = choose_phase_for_env(w, fs, lines, mains_hz, mains_bw, fmax_env)
else:
    idx = {"A":0,"B":1,"C":2}[ph]
f_env, P_env = envelope_psd(w[:, idx], fs, fmax=fmax_env, mains_hz=mains_hz, bw=mains_bw)


# вертикальные линии
fig_env = go.Figure()
fig_env.add_trace(go.Scatter(x=f_env, y=P_env, name="PSD огибающей", mode="lines"))
for k, fc in {"1×RPM":lines["fr"],"2×RPM":2*lines["fr"],"FTF":lines["FTF"],"BPFO":lines["BPFO"],"BPFI":lines["BPFI"],"BSF":lines["BSF"]}.items():
    fig_env.add_vline(x=fc, line_dash="dot", line_color="gray", annotation_text=k, annotation_position="top right")
fig_env.update_layout(height=280, xaxis_title="Гц", yaxis_title="Мощность (огибающая)")
st.plotly_chart(fig_env, use_container_width=True)

# рекомендация системы (тип, severity, уверенность, светофор)
rec_type, rec_sev, rec_conf, rec_flag, why_df = recommend_by_rules(f_env, P_env, lines)

st.markdown(
    f"**Рекомендация системы:** {to_ru(rec_type)} • Severity ≈ **{rec_sev:.1f}** • "
    f"Уверенность ≈ **{rec_conf:.2f}** • Статус: "
    + (":red_circle: **RED**" if rec_flag == "RED"
       else ":orange_circle: **ORANGE**" if rec_flag == "ORANGE"
       else ":green_circle: **GREEN**")
)

with st.expander("Почему так решено? (энергии опорных полос)", expanded=False):
    st.dataframe(why_df, use_container_width=True)

auto_fill = st.checkbox("Автозаполнить поля по рекомендации", value=True)

# --- форма разметки (русский UI) ---
with st.form("label_form", border=True):
    c1, c2, c3 = st.columns(3)
    opts_ru = [to_ru(x) for x in LABELS_EN]
    default_idx = LABELS_EN.index(rec_type) if rec_type in LABELS_EN else 0

    with c1:
        y_type_ru = st.selectbox("Тип", opts_ru, index=default_idx, disabled=auto_fill)
    with c2:
        severity = st.slider("Severity (0–100)", 0.0, 100.0,
                             float(np.round(rec_sev, 1)), disabled=auto_fill)
    with c3:
        confidence = st.slider("Уверенность (0–1)", 0.0, 1.0,
                               float(np.round(rec_conf, 2)), step=0.01, disabled=auto_fill)

    notes = st.text_input("Заметки", value="")
    y_defect_calc = (rec_type != "normal") or (rec_sev >= 20)
    st.caption("Светофор: RED ≥ 80, ORANGE ≥ 60, иначе GREEN. «Есть дефект?» = тип ≠ норма или severity ≥ 20.")
    colb = st.columns([1, 1, 2])
    with colb[0]:
        btn_accept = st.form_submit_button("✅ Принять рекомендацию", use_container_width=True)
    with colb[1]:
        btn_save = st.form_submit_button("💾 Сохранить (как указано выше)", use_container_width=True)

    if btn_accept or btn_save:
        # берём рекомендованное, если авто-заполнение включено или нажата «Принять»
        y_type_final = rec_type if (auto_fill or btn_accept) else to_en(y_type_ru)
        sev_final = float(rec_sev if (auto_fill or btn_accept) else severity)
        conf_final = float(rec_conf if (auto_fill or btn_accept) else confidence)
        y_defect_final = int((y_type_final != "normal") or (sev_final >= 20))

        df_lab = load_labels()
        rec = {
            "win_id": window_id(fsel, i0, i1),
            "file": os.path.basename(fsel),
            "i0": i0, "i1": i1, "fs": fs,
            "t0": i0 / fs, "t1": i1 / fs,
            "y_defect": y_defect_final,
            "y_type": y_type_final,
            "severity": sev_final,
            "confidence": conf_final,
            "notes": notes,
            "rpm": float(rpm), "Z": int(Z), "dmm": float(dmm), "Dmm": float(Dmm), "theta": int(theta)
        }
        df_lab = upsert_label(df_lab, rec)
        st.success(
            f"Сохранено: {rec['win_id']} → {to_ru(y_type_final)}, severity {sev_final:.1f}, conf {conf_final:.2f}. "
            f"Всего меток: {len(df_lab)}"
        )
        st.rerun()

with st.expander("⚙️ Авторазметить все TOP-N кандидатов по рекомендации", expanded=False):
    if st.button("🚀 Запустить авторазметку и сохранить"):
        df_lab = load_labels()
        saved = 0
        for idx in order:
            s0, s1 = spans[idx]
            wloc = x[s0:s1]
            # быстрая оценка по фазе A (при желании выбери лучшую фазу по энергии)
            f_b, P_b = envelope_psd(wloc[:,0], fs, fmax=fmax_env, mains_hz=mains_hz, bw=mains_bw)
            rtype, rsev, rconf, rflag, _ = recommend_by_rules(f_b, P_b, lines)
            rec = {
                "win_id": window_id(fsel, s0, s1),
                "file": os.path.basename(fsel),
                "i0": s0, "i1": s1, "fs": fs,
                "t0": s0 / fs, "t1": s1 / fs,
                "y_defect": int((rtype != "normal") or (rsev >= 20)),
                "y_type": rtype,
                "severity": float(rsev),
                "confidence": float(rconf),
                "notes": "auto",
                "rpm": float(rpm), "Z": int(Z), "dmm": float(dmm), "Dmm": float(Dmm), "theta": int(theta)
            }
            df_lab = upsert_label(df_lab, rec)
            saved += 1
        st.success(f"Готово: сохранено {saved} авто-меток.")

# ==== [PATCH 2] UI: пакетные автодобавления ====

with st.expander("🧪 Авто-НОРМА (добавить спокойные окна по выбранным CSV)", expanded=False):
    files_norm = st.multiselect("Файлы для авто-нормы", files, default=[fsel], key="auto_norm_files")
    n_per_file = st.number_input("Сколько «норма» окон на файл", 1, 100, 15, 1)
    sev_thr = st.slider("Макс. severity для нормы", 0.0, 40.0, 10.0, 1.0)
    strength_thr = st.slider("Макс. доля энергии подшипниковых полос", 0.0, 0.20, 0.05, 0.01)
    contam = st.slider("Contamination для IsolationForest", 0.01, 0.20, 0.05, 0.01)
    if st.button("➕ Добавить авто-норму"):
        total_saved = 0
        for p in files_norm:
            total_saved += autolabel_normals_for_file(
                p, fs_out=fs_out, win_sec=win_sec, overlap=OVERLAP,
                n_per_file=int(n_per_file), sev_thr=float(sev_thr),
                strength_thr=float(strength_thr), contam=float(contam),
                fmax=fmax_env
            )
        st.success(f"Готово: добавлено {total_saved} «норма»-окон по {len(files_norm)} файлам.")

with st.expander("🔎 Автопоиск кандидатов по типам (BPFO/BPFI/BSF/FTF/1×/2×RPM)", expanded=False):
    files_types = st.multiselect("Файлы для поиска", files, default=[fsel], key="auto_type_files")
    labels_sel = st.multiselect(
        "Какие типы искать",
        ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"],
        default=["BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
    )
    top_k = st.number_input("Top-K окон на тип и файл", 1, 50, 10, 1)
    # в UI
    min_score = st.slider("Мин. относительная энергия в целевой полосе",
                          0.0, 0.01, 0.0005, 0.0001, key="type_min_score", format="%.4f")

    if st.button("🚀 Найти и сохранить кандидатов по типам"):
        total_saved = 0
        for p in files_types:
            total_saved += autolabel_types_for_file(
                p, fs_out=fs_out, win_sec=win_sec, overlap=OVERLAP,
                labels=tuple(labels_sel), top_k=int(top_k), min_score=float(min_score),
                fmax=fmax_env
            )
        st.success(f"Готово: сохранено {total_saved} авто-меток по типам (в {len(files_types)} файлах).")
# ==== [END PATCH 2] ====

# ==== [PATCH 3] ВЕСЬ ДАТАСЕТ: авто-норма + кандидаты по типам + статистика ====

with st.expander("🧰 Авторазметить ВЕСЬ датасет (все CSV)", expanded=False):
    st.caption("Запустит авто-норму и поиск по типам для КАЖДОГО файла из data/raw/*.csv. "
               "Метки upsert-ятся: существующие окна обновятся, новые добавятся.")

    fmax_env_all = st.slider("Макс. частота огибающей, Гц (ВЕСЬ датасет)", 200.0, 400.0, 320.0, 10.0, key="all_fmax")
    files_all = files  # все CSV из data/raw/
    n_norm = st.number_input("Сколько НОРМА-окон на файл", 1, 100, 15, 1, key="all_n_norm")
    sev_thr_all = st.slider("Макс. severity для НОРМЫ", 0.0, 40.0, 10.0, 1.0, key="all_sev_thr")
    strength_thr_all = st.slider("Макс. доля подшипниковых полос (НОРМА)", 0.0, 0.30, 0.05, 0.01, key="all_str_thr")
    contam_all = st.slider("IsolationForest contamination", 0.01, 0.20, 0.05, 0.01, key="all_contam")

    labels_all = st.multiselect(
        "Какие типы искать",
        ["BPFO","BPFI","BSF","FTF","imbalance","misalignment"],
        default=["BPFO","BPFI","BSF","FTF","imbalance","misalignment"],
        key="all_labels"
    )
    topk_all = st.number_input("Top-K окон на тип и файл", 1, 50, 10, 1, key="all_topk")

    min_score_all = st.slider(
        "Мин. относительная энергия в целевой полосе",
        0.0, 0.01, 0.0005, 0.0001, key="dataset_min_score", format="%.4f"
    )
    # Параметры обработки для всего датасета
    cA, cB = st.columns(2)
    with cA:
        fs_out_all = st.select_slider("Fs, Гц (ВЕСЬ датасет)", [2000, 3200, 5120], value=fs_out)
        win_sec_all = st.select_slider("Окно, с (ВЕСЬ датасет)", [0.5, 1.0, 2.0], value=win_sec)
    with cB:
        mains_hz_all = st.select_slider("Частота сети, Гц (ВЕСЬ датасет)", options=[50.0, 60.0], value=mains_hz)
        mains_bw_all = st.slider("Полоса вокруг сети, Гц (ВЕСЬ датасет)", 2.0, 20.0, mains_bw, 0.5, format="%.1f")

    # Геометрия подшипника для всего датасета
    g1, g2, g3, g4, g5 = st.columns(5)
    rpm_all = g1.number_input("RPM (ВЕСЬ)", value=int(rpm), step=10)
    Z_all = g2.number_input("Z (ВЕСЬ)", value=int(Z), step=1)
    dmm_all = g3.number_input("d, мм (ВЕСЬ)", value=float(dmm), step=0.01, format="%.2f")
    Dmm_all = g4.number_input("D, мм (ВЕСЬ)", value=float(Dmm), step=0.1, format="%.1f")
    theta_all = g5.slider("θ, ° (ВЕСЬ)", 0, 30, int(theta))

    if st.button("🧨 Разметить ВЕСЬ датасет (все файлы)"):
        prog = st.progress(0)
        total_saved_norm, total_saved_types = 0, 0
        for idx, p in enumerate(files_all):
            try:
                total_saved_norm += autolabel_normals_for_file(
                    p, fs_out=fs_out_all, win_sec=win_sec_all, overlap=OVERLAP,
                    n_per_file=int(n_norm), sev_thr=float(sev_thr_all),
                    strength_thr=float(strength_thr_all), contam=float(contam_all),
                    rpm=rpm_all, Z=int(Z_all), dmm=float(dmm_all), Dmm=float(Dmm_all), theta=int(theta_all),
                    mains_hz=mains_hz_all, mains_bw=mains_bw_all, fmax=fmax_env_all
                )

                total_saved_types += autolabel_types_for_file(
                    p, fs_out=fs_out_all, win_sec=win_sec_all, overlap=OVERLAP,
                    labels=tuple(labels_all), top_k=int(topk_all), min_score=float(min_score_all),
                    rpm=rpm_all, Z=int(Z_all), dmm=float(dmm_all), Dmm=float(Dmm_all), theta=int(theta_all),
                    mains_hz=mains_hz_all, mains_bw=mains_bw_all, fmax=fmax_env_all
                )
            except Exception as e:
                st.warning(f"ТИПЫ: {os.path.basename(p)} — пропущен ({e})")
            prog.progress((idx + 1) / max(1, len(files_all)))

        st.success(f"Готово. Добавлено НОРМ: {total_saved_norm}, кандидатов по типам: {total_saved_types}.")

        # ---- СТАТИСТИКА ПО ТОЛЬКО ЧТО ОБРАБОТАННЫМ ФАЙЛАМ ----
        df_lab = load_labels()
        proc_files = [os.path.basename(p) for p in files_all]
        sub = df_lab[df_lab["file"].isin(proc_files)].copy()

        st.markdown("#### 📊 Статистика по авторазметке")
        st.write(f"Всего меток по обработанным файлам: **{len(sub)}**  "
                 f"(файлов: {len(proc_files)})")

        # распределение по типам
        per_type = sub["y_type"].value_counts().reindex(
            ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
        ).fillna(0).astype(int)
        st.markdown("**По типам (шт.)**")
        st.dataframe(per_type.rename("count").to_frame(), use_container_width=True)

        # светофор по severity
        sev_bins = pd.cut(sub["severity"], [-1, 60, 80, 100],
                          labels=["GREEN(<60)","ORANGE[60–80)","RED(≥80)"])
        by_flag = sev_bins.value_counts().reindex(["GREEN(<60)","ORANGE[60–80)","RED(≥80)"]).fillna(0).astype(int)
        st.markdown("**Светофор по severity**")
        st.dataframe(by_flag.rename("count").to_frame(), use_container_width=True)

        # топ по файлам
        st.markdown("**Топ по количеству меток на файл**")
        per_file = sub.groupby("file")["win_id"].count().sort_values(ascending=False)
        st.dataframe(per_file.rename("count").to_frame().head(20), use_container_width=True)

        # проверка готовности к обучению
        min_per_class = st.number_input("Порог готовности: минимум примеров на класс", 5, 200, 30, 5, key="all_minpercls")
        need = {c: int(per_type.get(c, 0)) for c in ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]}
        lack = {c: n for c, n in need.items() if n < min_per_class}
        if len(lack) == 0:
            st.success("✅ Баланс классов достаточен для старта обучения.")
        else:
            st.warning(f"⚠️ Не хватает примеров: {lack}. "
                       f"Повышай Top-K, понижайте порог энергии или прогоните дополнительные файлы.")
# ==== [END PATCH 3] ====

def estimate_fr_from_env(f, P, rpm_hint, rel_tol=0.05):
    """Ищем пик около fr_hint = rpm_hint/60 в пределах ±rel_tol."""
    fr_hint = rpm_hint / 60.0
    m = (f >= fr_hint*(1-rel_tol)) & (f <= fr_hint*(1+rel_tol))
    if not np.any(m):
        return None
    return float(f[m][np.argmax(P[m])])  # Гц

def percentile_scores_over_dataset(files, fs_out, win_sec, overlap, mains_hz, mains_bw,
                                   rpm_hint, Z, dmm, Dmm, theta, fmax):

    rows = []
    lines_hint = bearing_lines(rpm_hint, Z, dmm, Dmm, theta)
    for p in files:
        x, fs = load_and_resample(p, fs_out)
        for s, w in sliding_windows(x, fs, win_sec, overlap):
            idx = choose_phase_for_env(w, fs, lines_hint, mains_hz, mains_bw, fmax)
            f, P = envelope_psd(w[:, idx], fs, fmax=fmax, mains_hz=mains_hz, bw=mains_bw)
            fr_est = estimate_fr_from_env(f, P, rpm_hint)   # Гц или None
            rpm_use = float(fr_est*60.0) if fr_est is not None else float(rpm_hint)
            lines = bearing_lines(rpm_use, Z, dmm, Dmm, theta)
            total = _area(f, P) + 1e-12
            get = lambda fc, h=3: band_energy(f, P, fc, bw=1.0, harmonics=h) / total
            rows.append({
                "file": os.path.basename(p),
                "BPFO": get(lines["BPFO"], 3),
                "BPFI": get(lines["BPFI"], 3),
                "BSF":  get(lines["BSF"],  3),
                "FTF":  get(lines["FTF"],  3),
                "1x":   get(lines["fr"],   2),
                "2x":   get(2*lines["fr"], 2),
            })
    return pd.DataFrame(rows)


def sideband_snr(f0, P0, f_center, f_sb, w=1.0):
    """Отношение энергии в боковых полосах (f_center±f_sb) к ближайшему фону."""
    sb = 0.0
    for k in (-1, +1):
        m = (f0 >= f_center + k*f_sb - w) & (f0 <= f_center + k*f_sb + w)
        if np.any(m): sb += _area(f0, P0, m)
    # фон рядом, но вне ±w
    b = (f0 >= f_center-5*w) & (f0 <= f_center+5*w) & ~((f0 >= f_center-w) & (f0 <= f_center+w))
    noise = _area(f0, P0, b) / (np.sum(b) + 1e-9)
    return sb / (noise + 1e-12)

with st.expander("🩺 Диагностика датасета: есть ли BPFO/BPFI/BSF/FTF?", expanded=False):
    colm1, colm2 = st.columns(2)
    with colm1:
        mains_hz_diag = st.select_slider("Частота сети, Гц", options=[50.0, 60.0], value=50.0, key="diag_mhz")
    with colm2:
        mains_bw_diag = st.slider("Полоса вокруг сети, Гц", 4.0, 20.0, 10.0, 0.5, key="diag_mbw")
    fmax_diag = st.slider("Макс. частота огибающей, Гц (диаг.)", 200.0, 400.0, 320.0, 10.0, key="diag_fmax")

    files_all = list_files()
    if st.button("Посчитать персентили по всем CSV"):
        df = percentile_scores_over_dataset(files_all, fs_out, win_sec, OVERLAP,
                                            mains_hz_diag, mains_bw_diag, rpm, Z, dmm, Dmm, theta, fmax_diag)
        q = df[["BPFO","BPFI","BSF","FTF","1x","2x"]].quantile([0.5,0.9,0.95,0.99]).T
        q.columns = ["p50","p90","p95","p99"]
        st.markdown("**Относительная энергия по полосам (персентили):**")
        st.dataframe(q.style.format("{:.4f}"), use_container_width=True)
        for lab in ["BPFO","BPFI","BSF","FTF"]:
            top = df.nlargest(10, lab)[["file", lab]]
            st.write(f"Топ по {lab}:")
            st.dataframe(top.rename(columns={lab:"score"}).style.format({"score":"{:.4f}"}),
                         use_container_width=True)


# показать существующую метку (если есть)
df_lab = load_labels()
wid = window_id(fsel, i0, i1)
if wid in set(df_lab["win_id"]):
    st.caption("Метка уже существует:")
    st.dataframe(df_lab[df_lab["win_id"]==wid], use_container_width=True)
