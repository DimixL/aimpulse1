# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="ИИмпульс", layout="wide")

# ---- AUTH --- простой fallback ----
USE_ADV_AUTH = False
name = "Инженер"

try:
    import streamlit_authenticator as stauth
    import toml
    config = toml.load("config_auth.toml")
    authenticator = stauth.Authenticate(
        config['credentials'], config['cookie']['name'],
        config['cookie']['key'], config['cookie']['expiry_days']
    )
    name, auth_status, username = authenticator.login('Вход', 'main')
    if not auth_status:
        if auth_status is False:
            st.error('Неверный логин/пароль')
        st.stop()
    authenticator.logout('Выйти', 'sidebar')
    USE_ADV_AUTH = True
except ModuleNotFoundError:
    # простой вход без внешнего пакета
    st.sidebar.subheader("Вход")
    st.session_state.setdefault("auth_ok", False)
    if not st.session_state["auth_ok"]:
        user = st.sidebar.text_input("Логин", value="user1")
        pwd  = st.sidebar.text_input("Пароль", type="password", value="admin")
        if st.sidebar.button("Войти"):
            if user == "user1" and pwd == "admin":
                st.session_state["auth_ok"] = True
            else:
                st.sidebar.error("Неверный логин/пароль")
        if not st.session_state["auth_ok"]:
            st.stop()
    if st.sidebar.button("Выйти"):
        st.session_state["auth_ok"] = False
        st.rerun()

# --- Sidebar: профиль / справка ---
display_name = name if USE_ADV_AUTH else st.session_state.get("user_name", "Иван Иванов")
with st.sidebar:
    st.markdown(f"### 👤 {display_name}")
    try:
        # Streamlit 1.30+ — есть popover
        with st.popover("Профиль и справка"):
            st.write("""
**Имя:** {dn}  
**Роль:** Инженер-диагност  
**Контакты:** ivan.ivanov@example.com

— Справка:
- Аларм = severity ≥ 80 или p(defect) ≥ 0.8  
- Предупреждение = severity ≥ 50 или p(defect) ≥ 0.5  
- Источники дефектов — усреднение `proba` по окнам
            """.format(dn=display_name))
    except Exception:
        # На старых версиях можно expander
        with st.expander("Профиль и справка"):
            st.write("Имя: " + display_name)
            st.write("Роль: Инженер-диагност")
            st.write("Контакты: ivan.ivanov@example.com")
            st.write("Аларм: severity ≥ 80 или p(defect) ≥ 0.8")
            st.write("Предупреждение: severity ≥ 50 или p(defect) ≥ 0.5")


# --- PAGES --- скорректированы, для видео - часть страниц в проработке
page = st.sidebar.radio(
    "Разделы",
    ["ИИмпульс - Сводка", "ТОиР • Журнал поломок", "ИИ-предсказание (MVP)"]
)

if page.startswith("ИИмпульс - Сводка"):
    st.title("ИИмпульс — Сводка")
    st.caption("KPI, распределения и быстрый просмотр сырого сигнала по выбранному двигателю")

    import glob, os, numpy as np, pandas as pd
    from features import read_csv_3phase, decimate
    FS_RAW = 25600

    files = sorted(glob.glob("data/raw/*.csv"))
    if not files:
        st.warning("Положите CSV в папку data/raw/ и вернитесь на эту страницу.")
        st.stop()

    # --- параметры расчёта (для сводки и графиков) ---
    colp = st.columns(3)
    with colp[0]:
        fs_out = st.select_slider("Частота обработки (Гц)", options=[2000, 3200, 5120], value=3200)
    with colp[1]:
        win_sec = st.select_slider("Окно, сек", options=[0.5, 1.0, 2.0], value=1.0)
    with colp[2]:
        thresh_warn = st.slider("Порог предупреждения (severity / p)", 0, 100, 80, step=5)
    # --- сводная статистика по всем файлам (кэшируем) ---
    @st.cache_data(show_spinner=True)
    def summarize(files, fs_out, win_sec, warn_thr):
        try:
            from inference import predict_csv
        except Exception:
            return None  # инференса нет — покажем заглушки

        alarms = 0
        warns = 0
        per_source = np.zeros(6, dtype=float)  # распределение типов
        per_device = {}  # агрегаты по файлам

        for f in files:
            try:
                preds = predict_csv(f, fs_out=fs_out, win_sec=win_sec, overlap=0.0)
                df = pd.DataFrame(preds)
            except Exception:
                continue

            # критерии
            is_alarm = (df["severity"] >= 80) | (df["p_fault"] >= 0.8)
            is_warn  = (df["severity"] >= warn_thr) | (df["p_fault"] >= (warn_thr/100.0))

            alarms += int(is_alarm.sum())
            warns  += int(is_warn.sum())

            # распределение источников дефектов (по окнам, где есть предупреждение)
            if "proba" in df.columns:
                probs = np.array(df.loc[is_warn, "proba"].tolist()) if is_warn.any() else np.empty((0,6))
                if probs.size:
                    per_source += probs.mean(axis=0)  # мягкое усреднение

            per_device[os.path.basename(f)] = dict(
                alarms=int(is_alarm.sum()),
                warns=int(is_warn.sum()),
                severity_mean=float(df["severity"].mean() if "severity" in df else 0.0),
                p_fault_mean=float(df["p_fault"].mean() if "p_fault" in df else 0.0),
            )

        return dict(alarms=alarms, warns=warns, per_source=per_source, per_device=per_device)


    # стало
    calc_kpi = st.toggle(
        "Посчитать KPI по всем файлам",
        value=False,
        help="Запускает инференс predict_csv по всем CSV (может быть долго). Если выключено — покажем заглушки."
    )
    summary = summarize(files, fs_out, win_sec, thresh_warn) if calc_kpi else None

    # --- KPI (заглушки, если инференса нет) ---
    c1, c2, c3, c4 = st.columns(4)
    if summary is None:
        # нет артефактов модели — показываем заглушки
        c1.metric("Алармы (severity≥80 / p≥0.8)", "8")
        c2.metric("Предупреждения (≥"+str(thresh_warn)+")", "4")
        # «из 1С ТОИР» — заглушка
        repairs = st.session_state.get("repair_count", 3)
        c3.metric("Устройств в ремонте (1С ТОИР)", repairs)
        c4.metric("Двигателей на мониторинге", len(files))
    else:
        c1.metric("Алармы (severity≥80 / p≥0.8)", f"{summary['alarms']}")
        c2.metric("Предупреждения (≥"+str(thresh_warn)+")", f"{summary['warns']}")
        # «из 1С ТОИР» — сейчас заглушка, позже сюда коннектор
        repairs = st.session_state.get("repair_count", 3)
        c3.metric("Устройств в ремонте (1С ТОИР)", repairs)
        c4.metric("Двигателей на мониторинге", len(files))

        # круговая диаграмма по источникам дефектов
        st.subheader("Распределение вероятных источников дефектов (усреднение по окнам с предупреждениями)")
        labels = ["bearing_outer","bearing_inner","rolling","cage","imbalance","misalignment"]
        vals = summary["per_source"]
        if vals.sum() > 0:
            pie_df = pd.DataFrame({"source": labels, "value": vals / (vals.sum() + 1e-9)})
            fig_pie = px.pie(pie_df, names="source", values="value", hole=0.45)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("Нет предупреждений → распределение не рассчитано.")

    st.divider()

    # --- Быстрый просмотр сырого сигнала по выбранному двигателю ---
    st.subheader("Сырые сигналы (фрагмент) по выбранному двигателю")
    import numpy as np
    device_file = st.selectbox("Двигатель (CSV)", files, index=0, format_func=lambda p: os.path.basename(p))
    show_sec = st.slider("Длина фрагмента (сек)", 5, 30, 10)

    x = read_csv_3phase(device_file)
    factor = max(1, FS_RAW // fs_out)
    if factor > 1:
        x = decimate(x, factor)
    fs = FS_RAW // factor
    n = len(x); dur = n / fs

    # позиция фрагмента
    pos = st.slider("Позиция начала (сек)", 0.0, max(0.0, dur - show_sec), 0.0, 0.5)
    i0 = int(pos * fs); i1 = min(i0 + int(show_sec * fs), n)
    frag = x[i0:i1]

    # downsample для рисования
    max_points = 8000
    step = max(1, (i1 - i0) // max_points)
    t = np.arange(i0, i1, step) / fs
    frag_ds = frag[::step]

    fig = go.Figure()
    for i, name in enumerate(["A","B","C"]):
        fig.add_trace(go.Scattergl(x=t, y=frag_ds[:, i], mode="lines", name=name, line=dict(width=1)))
    fig.update_layout(height=320, xaxis_title="Время, с", yaxis_title="Ток, A",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True)


elif page.startswith("ИИмпульс (Главная)"):
    st.title("ИИмпульс — мониторинг состояния")
    st.write("Здесь будет сводка по последним анализам, KPI модели и тренды.")
    st.info("Для теста перейдите на вкладку «Анализ CSV», загрузите файл и получите прогноз.")
elif page.startswith("Обзор датасета"):
    st.title("Обзор датасета (38 файлов)")
    st.caption("Листаем 38 CSV: время-ряд, спектр, PSD, спектрограмма и быстрый поиск подозрительных окон")

elif page.startswith("ИИ-предсказание"):
    # ===== ЗАГОЛОВОК =====
    st.title("🔮 ИИ-диагностика по CSV (МVP)")
    st.caption("Загрузите CSV (3 колонки тока A,B,C) — получим вероятность дефекта, тип, severity и TTF.")

    # --- чтобы подхватились наши локальные модули (inference_v2, features, mvp_model_runtime) ---
    import os, sys, io
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    sys.path.append(os.path.dirname(__file__))
    from inference_v2 import load_predictor

    # ===== МОДЕЛЬ (кэш ресурса) =====
    @st.cache_resource(show_spinner=False)
    def _load_predictor():
        return load_predictor(model_dir="models")
    predictor = _load_predictor()

    # ===== САЙДБАР: НАСТРОЙКИ =====
    with st.sidebar:
        st.header("Параметры анализа (MVP)")
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


elif page.startswith("ТОиР"):
    st.title("ТОиР • Журнал поломок")
    st.caption("Загрузи журнал или работай на демо-данных. Фильтры — слева.")

    import io, math, re, random
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    REQUIRED_COLS = [
        "ID записи", "Дата обнаружения", "Время обнаружения", "Цех / участок",
        "Оборудование / двигатель", "Инвентарный № / серийный №", "Тип оборудования",
        "Тип поломки", "Описание неисправности", "Критичность", "Создана заявка",
        "Номер заявки", "Ответственный за устранение", "Дата/время создания заявки",
        "Срок устранения (план)", "Дата фактического устранения",
        "Время простоя", "Статус", "Причина поломки", "Принятые меры",
        "Исполнитель ремонта", "Стоимость ремонта / запчастей", "Комментарии"
    ]


    # ---------- helpers ----------
    def _to_bool(x):
        if isinstance(x, str):
            x = x.strip().lower()
        return x in (True, 1, "1", "true", "да", "y", "yes", "ок", "истина")


    @st.cache_data(show_spinner=False)
    def load_uploaded(file):
        if file is None:
            return None
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine="openpyxl")
            return df
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            return None


    @st.cache_data(show_spinner=False)
    def make_demo(n_rows=1000, days=180, seed=42):
        random.seed(seed);
        np.random.seed(seed)
        today = datetime.now().date()
        shops = [f"Цех {i}" for i in range(1, 9)]
        types = ["двигатель", "станок", "компрессор", "насос", "редуктор"]
        eqs = [f"Двигатель-{i:03d}" for i in range(40, 120)]
        kinds = ["механическая", "электрическая", "гидравлика", "ПО", "смазка"]
        reasons = ["износ", "ошибка оператора", "перегрузка", "дефект детали", "вибрация", "загрязнение"]
        crits = ["высокая", "средняя", "низкая"]
        statuses = ["В работе", "Ожидает запчастей", "Устранена"]
        resp = ["Иванов", "Петров", "Сидоров", "Смирнов", "Кузнецов"]
        execs = ["АО РемонтСервис", "ООО ТехСаппорт", "ИП Механик", "Цеховая бригада"]

        rows = []
        for i in range(n_rows):
            d0 = today - timedelta(days=int(np.random.beta(2, 6) * days))
            # вероятность наличия времени
            t0 = (datetime(2000, 1, 1, 8, 0) + timedelta(
                minutes=int(np.random.rand() * 600))).time() if np.random.rand() < 0.8 else None
            sh = random.choice(shops)
            tp = random.choice(types)
            eq = random.choice(eqs)
            kind = np.random.choice(kinds, p=[0.36, 0.32, 0.1, 0.12, 0.1])
            crit = np.random.choice(crits, p=[0.25, 0.5, 0.25])
            stt = np.random.choice(statuses, p=[0.25, 0.15, 0.60])
            created_dt = datetime.combine(d0, t0 or datetime.min.time()) + timedelta(hours=np.random.uniform(0, 6))
            plan_dt = created_dt + timedelta(hours=np.random.uniform(4, 72))
            closed = (stt == "Устранена") and (np.random.rand() < 0.9)
            fix_dt = created_dt + timedelta(hours=np.random.uniform(2, 48)) if closed else pd.NaT
            downtime = float(np.round(np.random.uniform(0.5, 24), 2)) if closed else np.nan
            cost = float(np.round(np.exp(np.random.normal(6, 0.9)) / 100, 2)) if closed else np.nan  # ~логнорм
            rows.append({
                "ID записи": i + 1,
                "Дата обнаружения": d0,
                "Время обнаружения": t0,
                "Цех / участок": sh,
                "Оборудование / двигатель": eq,
                "Инвентарный № / серийный №": f"INV-{10000 + i}",
                "Тип оборудования": tp,
                "Тип поломки": kind,
                "Описание неисправности": f"Симптомы {kind} на {eq}",
                "Критичность": crit,
                "Создана заявка": "Да" if np.random.rand() < 0.85 else "Нет",
                "Номер заявки": f"REQ-{200000 + i}" if np.random.rand() < 0.85 else "",
                "Ответственный за устранение": random.choice(resp),
                "Дата/время создания заявки": created_dt,
                "Срок устранения (план)": plan_dt,
                "Дата фактического устранения": fix_dt,
                "Время простоя": downtime,
                "Статус": stt,
                "Причина поломки": np.random.choice(reasons, p=[0.35, 0.15, 0.15, 0.1, 0.15, 0.10]),
                "Принятые меры": "Диагностика/ремонт/замена",
                "Исполнитель ремонта": random.choice(execs),
                "Стоимость ремонта / запчастей": cost,
                "Комментарии": ""
            })
        df = pd.DataFrame(rows)
        return df


    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        # убедимся, что все колонки есть
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error("Не хватает колонок: " + ", ".join(missing))
        # приведём типы
        df = df.copy()
        # даты/время
        df["Дата обнаружения"] = pd.to_datetime(df["Дата обнаружения"], errors="coerce").dt.date
        if "Время обнаружения" in df:
            df["Время обнаружения"] = pd.to_datetime(df["Время обнаружения"], errors="coerce").dt.time
        df["Дата/время создания заявки"] = pd.to_datetime(df["Дата/время создания заявки"], errors="coerce")
        df["Срок устранения (план)"] = pd.to_datetime(df["Срок устранения (план)"], errors="coerce")
        df["Дата фактического устранения"] = pd.to_datetime(df["Дата фактического устранения"], errors="coerce")
        # числа/булевые
        df["Создана заявка"] = df["Создана заявка"].map(_to_bool)
        for col in ["Время простоя", "Стоимость ремонта / запчастей"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # вычисляемые поля
        ts_detect = pd.to_datetime(df["Дата обнаружения"], errors="coerce")
        if "Время обнаружения" in df and df["Время обнаружения"].notna().any():
            # если есть время — добавим его
            tcomp = pd.to_datetime(df["Время обнаружения"].astype(str), errors="coerce").dt.time
            ts_detect = pd.to_datetime(df["Дата обнаружения"].astype(str) + " " + df["Время обнаружения"].astype(str),
                                       errors="coerce")
        df["__ts_detect"] = ts_detect
        # длительность устранения, ч
        mask_closed = df["Дата фактического устранения"].notna()
        delta = df.loc[mask_closed, "Дата фактического устранения"] - df.loc[mask_closed, "__ts_detect"]
        df["Длительность устранения, ч"] = np.nan
        df.loc[mask_closed, "Длительность устранения, ч"] = (delta.dt.total_seconds() / 3600.0).clip(lower=0)
        return df


    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        with st.sidebar:
            st.subheader("Фильтры журнала")
            min_d = pd.to_datetime(df["__ts_detect"]).min()
            max_d = pd.to_datetime(df["__ts_detect"]).max()
            d_from, d_to = st.date_input("Период (Дата обнаружения)", (min_d.date(), max_d.date()))
            shop = st.multiselect("Цех / участок", sorted(df["Цех / участок"].dropna().unique().tolist()))
            tpe = st.multiselect("Тип оборудования", sorted(df["Тип оборудования"].dropna().unique().tolist()))
            equip = st.multiselect("Оборудование / двигатель",
                                   sorted(df["Оборудование / двигатель"].dropna().unique().tolist()))
            kind = st.multiselect("Тип поломки", sorted(df["Тип поломки"].dropna().unique().tolist()))
            crit = st.multiselect("Критичность", ["высокая", "средняя", "низкая"])
            stat = st.multiselect("Статус", sorted(df["Статус"].dropna().unique().tolist()))
            only_closed = st.checkbox("Только закрытые инциденты", value=False)
            q = st.text_input("Поиск в описании/комментариях")

        m = (pd.to_datetime(df["__ts_detect"]).dt.date >= d_from) & (pd.to_datetime(df["__ts_detect"]).dt.date <= d_to)
        if shop: m &= df["Цех / участок"].isin(shop)
        if tpe:  m &= df["Тип оборудования"].isin(tpe)
        if equip: m &= df["Оборудование / двигатель"].isin(equip)
        if kind: m &= df["Тип поломки"].isin(kind)
        if crit: m &= df["Критичность"].isin(crit)
        if stat: m &= df["Статус"].isin(stat)
        if only_closed: m &= df["Дата фактического устранения"].notna()
        if q:
            ql = q.strip().lower()
            m &= (df["Описание неисправности"].astype(str).str.lower().str.contains(ql)) | \
                 (df["Комментарии"].astype(str).str.lower().str.contains(ql))
        return df[m].copy()


    def kpi_cards(df: pd.DataFrame):
        c1, c2, c3, c4 = st.columns(4)
        total = len(df)
        c1.metric("Поломок за период", f"{total}")
        mttr = df["Длительность устранения, ч"].dropna().mean()
        c2.metric("MTTR, ч", "—" if np.isnan(mttr) else f"{mttr:.1f}")
        # MTBF по оборудованию
        mtbf_vals = []
        for eq, g in df.sort_values("__ts_detect").groupby("Оборудование / двигатель"):
            times = pd.to_datetime(g["__ts_detect"]).values
            if len(times) >= 2:
                diffs = np.diff(times).astype("timedelta64[h]").astype(float)
                if len(diffs) > 0: mtbf_vals.append(diffs.mean())
        mtbf = np.mean(mtbf_vals) if mtbf_vals else np.nan
        c3.metric("MTBF, ч", "—" if (not mtbf_vals or np.isnan(mtbf)) else f"{mtbf:.0f}")
        # Доля «в срок»
        closed = df[df["Дата фактического устранения"].notna()].copy()
        if len(closed):
            ontime = (closed["Дата фактического устранения"] <= closed["Срок устранения (план)"]).mean() * 100
            c4.metric("Доля «в срок», %", f"{ontime:.0f}")
        else:
            c4.metric("Доля «в срок», %", "—")
        st.caption(
            "MTTR — среднее время устранения по закрытым инцидентам. MTBF — средний интервал между отказами по оборудованию с ≥2 событиями. «В срок» — сравнение факта и плана.")


    def export_buttons(df: pd.DataFrame, fname_prefix="journal_filtered"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать данные (CSV)", csv, file_name=f"{fname_prefix}.csv", mime="text/csv")
        try:
            import io
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as wr:
                df.to_excel(wr, index=False, sheet_name="data")
            st.download_button("Скачать данные (XLSX)", bio.getvalue(), file_name=f"{fname_prefix}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.caption("Для XLSX установите пакет openpyxl.")


    # ---------- загрузка/демо ----------
    with st.sidebar:
        st.subheader("Источник данных")
        upl = st.file_uploader("Загрузить журнал (CSV/XLSX)", type=["csv", "xlsx"], key="toir_upload")
    df_raw = load_uploaded(upl)
    if df_raw is None:
        df_raw = make_demo()

    df = normalize_df(df_raw)

    # ---------- фильтры + KPI ----------
    dff = apply_filters(df)
    kpi_cards(dff)

    # ---------- вкладки ----------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Общее", "Оборудование", "Типы поломок", "Сроки и эффективность", "Финансы", "Прогноз"]
    )

    with tab0:
        fr = st.radio("Шаг времени", ["День", "Неделя", "Месяц"], horizontal=True)
        code = {"День": "D", "Неделя": "W", "Месяц": "M"}[fr]
        ts = dff.set_index(pd.to_datetime(dff["__ts_detect"])).sort_index()
        if len(ts):
            trend = ts["ID записи"].resample(code).count().rename("count").to_frame()
            fig = px.line(trend, y="count", markers=True)
            fig.update_layout(height=320, xaxis_title="Дата", yaxis_title="Кол-во")
            st.plotly_chart(fig, use_container_width=True)
            # stacked bar по критичности
            gb = ts.groupby([pd.Grouper(freq=code), "Критичность"])["ID записи"].count().reset_index()
            fig2 = px.bar(gb, x="__ts_detect", y="ID записи", color="Критичность", barmode="stack")
            fig2.update_layout(height=320, xaxis_title="Дата", yaxis_title="Кол-во")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Нет данных для выбранных фильтров.")
        st.subheader("Последние инциденты")
        st.dataframe(dff.sort_values("__ts_detect", ascending=False).head(20), use_container_width=True)
        export_buttons(dff)

    with tab1:
        c = dff["Оборудование / двигатель"].value_counts().head(5).reset_index()
        c.columns = ["Оборудование / двигатель", "Кол-во"]
        fig = px.bar(c, x="Кол-во", y="Оборудование / двигатель", orientation="h")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        # heatmap цех × тип оборудования
        pv = dff.pivot_table(index="Тип оборудования", columns="Цех / участок", values="ID записи", aggfunc="count",
                             fill_value=0)
        fig2 = px.imshow(pv, text_auto=True, aspect="auto")
        st.plotly_chart(fig2, use_container_width=True)
        # суммарный простой
        down = dff.groupby("Оборудование / двигатель")["Время простоя"].sum().sort_values(ascending=False).head(10)
        fig3 = px.bar(down.reset_index(), x="Оборудование / двигатель", y="Время простоя")
        fig3.update_layout(height=320, xaxis_title="Оборудование", yaxis_title="Суммарный простой, ч")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        pie = dff["Тип поломки"].value_counts(normalize=True).reset_index()
        pie.columns = ["Тип поломки", "Доля"]
        fig = px.pie(pie, names="Тип поломки", values="Доля", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        sum_down = dff.groupby("Тип поломки")["Время простоя"].sum().reset_index()
        fig2 = px.bar(sum_down, x="Тип поломки", y="Время простоя")
        fig2.update_layout(height=320, yaxis_title="Суммарный простой, ч")
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Повторяемость дефектов (тип × причина)")
        tbl = dff.pivot_table(index="Тип поломки", columns="Причина поломки", values="ID записи", aggfunc="count",
                              fill_value=0)
        st.dataframe(tbl, use_container_width=True)

    with tab3:
        closed = dff[dff["Дата фактического устранения"].notna()].copy()
        if len(closed):
            ontime = (closed["Дата фактического устранения"] <= closed["Срок устранения (план)"]).mean()
            fig = px.bar(x=["В срок", "Просрочены"], y=[ontime * 100, (1 - ontime) * 100])
            fig.update_layout(height=300, yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)
            mttr_by = closed.groupby("Ответственный за устранение")["Длительность устранения, ч"].mean().sort_values()
            fig2 = px.bar(mttr_by.reset_index(), x="Ответственный за устранение", y="Длительность устранения, ч")
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)
            # «Гантт»
            g = closed.dropna(subset=["Дата/время создания заявки", "Дата фактического устранения"]).copy()
            if len(g):
                import plotly.express as px

                fig3 = px.timeline(
                    g, x_start="Дата/время создания заявки", x_end="Дата фактического устранения",
                    y="Оборудование / двигатель", color="Критичность", hover_data=["Статус", "Тип поломки"]
                )
                fig3.update_yaxes(autorange="reversed")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("Недостаточно дат для диаграммы Ганта.")
        else:
            st.info("Нет закрытых инцидентов для этой вкладки.")

    with tab4:
        cost_by_eq = dff.groupby("Оборудование / двигатель")["Стоимость ремонта / запчастей"].sum().sort_values(
            ascending=False).head(10)
        fig = px.bar(cost_by_eq.reset_index(), x="Оборудование / двигатель", y="Стоимость ремонта / запчастей")
        fig.update_layout(height=320, yaxis_title="Сумма, у.е.")
        st.plotly_chart(fig, use_container_width=True)
        cost_by_type = dff.groupby("Тип поломки")["Стоимость ремонта / запчастей"].sum().reset_index()
        fig2 = px.bar(cost_by_type, x="Тип поломки", y="Стоимость ремонта / запчастей")
        st.plotly_chart(fig2, use_container_width=True)
        # bubble: стоимость vs простой
        agg = dff.groupby("Тип поломки").agg(
            Стоимость=("Стоимость ремонта / запчастей", "sum"),
            Простой=("Время простоя", "sum"),
            Количество=("ID записи", "count")
        ).reset_index()
        fig3 = px.scatter(agg, x="Стоимость", y="Простой", size="Количество", color="Тип поломки",
                          hover_name="Тип поломки")
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        # простой прогноз количества поломок по оборудованию (скользящее среднее + экстра)
        eq_sel = st.selectbox(
            "Оборудование для прогноза",
            sorted(dff["Оборудование / двигатель"].dropna().unique())
        )

        sub = dff[dff["Оборудование / двигатель"] == eq_sel].copy()
        sub["__ts_detect"] = pd.to_datetime(sub["__ts_detect"])
        sub = sub.dropna(subset=["__ts_detect"])  # на всякий случай

        ts = sub.set_index("__ts_detect").sort_index()

        if len(ts):
            weekly = ts["ID записи"].resample("W").count().rename("count").to_frame()
            weekly["sma"] = weekly["count"].rolling(3, min_periods=1).mean()
            last = weekly["sma"].iloc[-1]
            fut = pd.DataFrame({"sma": [last] * 4},
                               index=pd.date_range(weekly.index.max() + pd.Timedelta(days=7),
                                                   periods=4, freq="W"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly["count"], mode="lines+markers", name="Факт/неделя"))
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly["sma"], mode="lines", name="SMA(3)"))
            fig.add_trace(go.Scatter(x=fut.index, y=fut["sma"], mode="lines", name="Прогноз",
                                     line=dict(dash="dash")))
            fig.update_layout(height=320, xaxis_title="Неделя", yaxis_title="Кол-во")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных по выбранному оборудованию.")
        # кандидаты на ТО
        st.subheader("Кандидаты на плановое ТО")
        risks = []
        for eq, g in dff.sort_values("__ts_detect").groupby("Оборудование / двигатель"):
            times = pd.to_datetime(g["__ts_detect"]).values
            if len(times) >= 3:
                diffs = np.diff(times).astype("timedelta64[h]").astype(float)
                risks.append({"Оборудование / двигатель": eq, "MTBF, ч": float(np.mean(diffs)), "Событий": len(g)})
        if risks:
            r = pd.DataFrame(risks).sort_values(["MTBF, ч", "Событий"], ascending=[True, False]).head(5)
            st.dataframe(r, use_container_width=True)
        else:
            st.caption("Недостаточно повторяющихся событий для итогового списка.")

    st.divider()
    st.caption(
        "Экспорт всей текущей выборки находится во вкладке «Общее». Для PNG-экспорта графиков можно установить пакет `kaleido`.")

else:
    st.title("Анализ CSV")
    fs_out = st.select_slider("Частота обработки (Гц)", options=[2000, 3200, 5120], value=3200)
    win = st.slider("Окно (сек)", 0.5, 2.0, 1.0, 0.5)
    up = st.file_uploader("Загрузите CSV (3 колонки тока)", type=["csv"])
    if up is not None:
        # показываем «сырое»
        raw = pd.read_csv(up)
        st.subheader("Сырые сигналы (фрагмент)")
        st.line_chart(raw.head(500))
        # сохраняем временно в память
        up.seek(0); data_bytes = up.read()
        tmp = io.BytesIO(data_bytes)
        # предсказания
        with st.spinner("Считаем..."):
            # сохранение во временный файл, если нужно: tmp_path = "/tmp/upload.csv"
            preds = predict_csv(io.BytesIO(data_bytes), fs_out=fs_out, win_sec=win, overlap=0.0)
        df = pd.DataFrame(preds)
        st.subheader("Таймлайн вероятностей и степени")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.t1, y=df.p_fault, name="p(defect)"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.t1, y=df.severity, name="severity (0..100)"))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Распределение по типам дефектов (среднее по файлу)")
        cls = ["bearing_outer","bearing_inner","rolling","cage","imbalance","misalignment"]
        avg = np.array(df['proba'].tolist()).mean(axis=0)
        st.bar_chart(pd.DataFrame({"p":avg}, index=cls))

        st.download_button("Скачать отчёт (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="report.csv")
