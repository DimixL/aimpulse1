# mvp_single.py
import io, os, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- чтобы подхватились наши локальные модули (inference_v2, features, mvp_model_runtime) ---
sys.path.append(os.path.dirname(__file__))

from inference_v2 import load_predictor

st.set_page_config(page_title="ИИ-диагностика CSV (MVP)", layout="wide")

# ===== ЗАГОЛОВОК =====
st.title("🔮 ИИ-диагностика по CSV (MVP)")
st.caption("Загрузите CSV (3 колонки тока A,B,C) — получим вероятность дефекта, тип, severity и TTF.")

# ===== МОДЕЛЬ (кэш ресурса) =====
@st.cache_resource(show_spinner=False)
def _load_predictor():
    return load_predictor(model_dir="models")
predictor = _load_predictor()

# ===== САЙДБАР: НАСТРОЙКИ =====
with st.sidebar:
    st.header("Параметры анализа")
    fs_out  = st.select_slider("Частота обработки, Гц", [2000, 3200, 5120], value=3200)
    win_sec = st.select_slider("Окно, сек", [0.5, 1.0, 2.0], value=1.0)
    overlap = st.select_slider("Перекрытие", [0.0, 0.25, 0.5], value=0.0)
    st.divider()
    st.header("Подшипник / RPM")
    rpm   = st.number_input("RPM", 500, 6000, 1770, step=10)
    Z     = st.number_input("Z (число тел качения)", 3, 20, 9, step=1)
    dmm   = st.number_input("d, мм (шарик/ролик)", value=7.94, step=0.01, format="%.2f")
    Dmm   = st.number_input("D, мм (делительная)", value=38.5, step=0.1, format="%.1f")
    theta = st.number_input("Угол контакта, °", value=0, step=1)
    st.caption(f"Порог детектора из модели: **{predictor.info.get('bin_threshold',0.5):.3f}**")

# ===== ФАЙЛ =====
up = st.file_uploader("Загрузите CSV с тремя колонками (A,B,C)", type=["csv"])
if not up:
    st.stop()

data_bytes = up.read()

# ===== ИНФЕРЕНС =====
try:
    with st.spinner("Считаем предсказания..."):
        df_pred, agg = predictor.predict_csv(
            io.BytesIO(data_bytes),
            fs_raw=25600, fs_out=fs_out,
            win_sec=win_sec, overlap=overlap,
            rpm=float(rpm), Z=int(Z), dmm=float(dmm), Dmm=float(Dmm), theta=int(theta)
        )
except Exception as e:
    st.error(f"Ошибка инференса: {e}")
    st.stop()

# ===== KPI / ВЕРДИКТ =====
thr = agg["thr_bin"]
verdict = "ДЕФЕКТ" if (agg["p_def_mean"] >= thr or agg["p_def_share_over_thr"] >= 0.2) else "НОРМА"
emoji = "🔴" if verdict == "ДЕФЕКТ" else "🟢"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Вердикт", f"{emoji} {verdict}")
k2.metric("Средняя p(defect)", f"{agg['p_def_mean']:.2f}")
k3.metric("Окон ≥ порога", f"{100*agg['p_def_share_over_thr']:.0f}%")
k4.metric("TTF до severity=80", "∞" if np.isinf(agg["ttf_to_80_sec"]) else f"{agg['ttf_to_80_sec']:.0f} c")

k5, k6, k7 = st.columns(3)
k5.metric("Класс файла (7)", agg["file_class7"])
k6.metric("Класс файла (3)", agg["file_class3"])
k7.metric("Severity avg / max", f"{agg['severity_mean']:.1f} / {agg['severity_max']:.1f}")

st.divider()

# ===== ГРАФИКИ =====
c1, c2 = st.columns(2)

with c1:
    st.subheader("Вероятность дефекта по времени")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_pred["t1"], y=df_pred["p_def"], mode="lines", name="p(defect)"))
    fig1.add_hline(y=thr, line_dash="dot", annotation_text="порог", annotation_position="top left")
    fig1.update_layout(height=280, xaxis_title="Время, c", yaxis_title="Вероятность")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Severity (0..100) по времени")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_pred["t1"], y=df_pred["severity"], mode="lines", name="severity"))
    fig2.add_hline(y=50, line_dash="dot", annotation_text="warning")
    fig2.add_hline(y=80, line_dash="dot", annotation_text="alarm")
    fig2.update_layout(height=280, xaxis_title="Время, c", yaxis_title="Severity")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Распределение типов дефектов (усреднение по файлу)")
labels7 = predictor.labels7 if hasattr(predictor, "labels7") else ["normal","BPFO","BPFI","BSF","FTF","imbalance","misalignment"]
p7 = np.array(agg["file_p7"])
df_p7 = pd.DataFrame({"class": labels7, "prob": p7})
fig_p7 = px.bar(df_p7, x="class", y="prob", text="prob")
fig_p7.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_p7.update_layout(height=280, yaxis_range=[0,1], margin=dict(t=20))
st.plotly_chart(fig_p7, use_container_width=True)

# ===== ТАБЛИЦА ОКОН =====
st.subheader("Окна с наибольшим риском")
top = df_pred.sort_values(["p_def","severity"], ascending=False).head(25)[["t0","t1","p_def","y_pred","severity"]]
st.dataframe(top, use_container_width=True, height=340)

# ===== ЭКСПОРТ =====
st.download_button(
    "⬇️ Скачать отчёт по окнам (CSV)",
    df_pred.to_csv(index=False).encode("utf-8"),
    file_name=f"report_{up.name}.csv",
    mime="text/csv"
)
