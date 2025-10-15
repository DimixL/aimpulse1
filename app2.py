# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import math
import html
from dataclasses import dataclass, field
from datetime import datetime
import glob, os, random
from typing import List, Optional, Literal

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*args, **kwargs):
        return None

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx  # noqa: F401 - optional helper
except Exception:
    pass

try:
    from inference_v2 import LABELS_ALL as _LABELS_ALL
except Exception:
    _LABELS_ALL = [
        "normal",
        "BPFO",
        "BPFI",
        "BSF",
        "FTF",
        "imbalance",
        "misalignment",
    ]

st.set_page_config(page_title="ИИмпульс", layout="wide")

st.markdown("""
<style>
/* Баннеры */
.big-banner{padding:18px 22px;border-radius:14px;display:flex;gap:14px;align-items:flex-start;margin:6px 0 14px;border:1px solid}
.big-banner .title{font-weight:800;font-size:22px;line-height:1.25;margin:0}
.big-banner .subtitle{margin:6px 0 0;font-size:14px;opacity:.85}
.big-banner.danger{background:#fee2e2;border-color:#fecaca;color:#7f1d1d}
.big-banner.ok{background:#e7f8ee;border-color:#c7f0d9;color:#064e3b}

/* Чипы */
.badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#F3F4F6;
       font-weight:600;font-size:13px;margin-right:8px}

/* KPI карточки под баннером */
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:6px 0 18px}
.kpi{padding:14px;border-radius:12px;background:#F8FAFC;border:1px solid #EFF2F6}
.kpi .label{font-size:12px;color:#64748B;text-transform:uppercase;letter-spacing:.02em}
.kpi .value{font-size:24px;font-weight:700;margin-top:4px}
@media (max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
""", unsafe_allow_html=True)

# --- Blue accent everywhere (sliders / radios / toggles / buttons) ---
st.markdown("""
<style>
/* 0) Жёстко задаём primary-цвет для Streamlit-темы */
:root{
  --primary-color:#2563EB !important;          /* главный акцент */
  --text-color:#111827;
}

/* 1) SLIDER: бегунок */
div[data-baseweb="slider"] [role="slider"]{
  background-color: var(--primary-color) !important;
  border-color: var(--primary-color) !important;
  box-shadow: 0 0 0 2px rgba(37,99,235,.22) !important;
}

/* 2) SLIDER: активная часть трека
   (покрываем разные версии: через структуру, классы BaseWeb и даже инлайн-стили по цвету) */
div[data-baseweb="slider"] > div > div > div[aria-hidden="true"],
div[data-baseweb="slider"] .Track,
div[data-baseweb="slider"] .css-1ldw2k6-Track,
div[data-baseweb="slider"] div[style*="rgb(255, 75, 75)"]{          /* ловим красный по умолчанию */
  background-color: var(--primary-color) !important;
}

/* 3) Значение над бегунком (обычно красное) — делаем синим */
div[data-testid="stSlider"] [data-baseweb="slider"] span,
div[data-testid="stSlider"] .stSliderValue,
div[data-baseweb="slider"] span[style*="rgb(255, 75, 75)"]{         /* если цвет пришёл инлайном */
  color: var(--primary-color) !important;
}

/* 4) Toggle / Switch */
div[role="switch"][aria-checked="true"]{
  background-color: var(--primary-color) !important;
  border-color: var(--primary-color) !important;
}

/* 5) Radio / Checkbox */
div[role="radio"][aria-checked="true"] > div,
div[role="checkbox"][aria-checked="true"] > div{
  background-color: var(--primary-color) !important;
  border-color: var(--primary-color) !important;
}

/* 6) Кнопки */
.stButton>button, .stDownloadButton>button{
  border-color: var(--primary-color) !important;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  background: rgba(37,99,235,.08) !important;
}
</style>
""", unsafe_allow_html=True)

# MONITOR: дополнительный CSS для карточек / уведомлений
st.markdown("""
<style>
.mon-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:10px 0 18px}
@media (max-width:1200px){.mon-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:768px){.mon-grid{grid-template-columns:repeat(1,minmax(0,1fr))}}
.mon-card{position:relative;background:#fff;border-radius:16px;border:1px solid #E5E7EB;padding:16px;box-shadow:0 6px 16px rgba(15,23,42,0.08);display:flex;flex-direction:column;gap:10px;min-height:200px}
.mon-card.alarm{background:#fee2e2;border-color:#fecaca;color:#7f1d1d}
.mon-card.warn{background:#fef3c7;border-color:#fde68a;color:#78350f}
.mon-card.ok{background:#e7f8ee;border-color:#c7f0d9;color:#064e3b}
.mon-card .mon-card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.mon-card-title{font-weight:700;font-size:20px}
.mon-badge{background:#f3f4f6;color:#4b5563;border-radius:999px;padding:4px 10px;font-size:11px;font-weight:600;text-transform:uppercase;white-space:nowrap}
.mon-badge.mon-badge--ghost{visibility:hidden}
.mon-card-equip{font-size:14px;font-weight:600;color:#1f2937}
.mon-card.alarm .mon-card-equip{color:#7f1d1d}
.mon-card.warn .mon-card-equip{color:#78350f}
.mon-card.ok .mon-card-equip{color:#064e3b}
.mon-card-shop{font-size:13px;color:rgba(55,65,81,0.9)}
.mon-card-viol{font-size:13px;color:rgba(17,24,39,0.92)}
.mon-pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 10px;margin:4px 4px 0 0;font-weight:600;font-size:12px;color:#111827;background:#E5E7EB}
.mon-pill.alarm{background:#ef4444;color:#fff}
.mon-pill.warn{background:#f59e0b;color:#fff}
.mon-pill.ok{background:#10b981;color:#fff}
.mon-ttf{font-size:12px;color:#b91c1c;font-style:italic}
.mon-card-caption{font-size:12px;color:#4b5563}
.mon-ai-holder{margin-top:auto;border:1px dashed rgba(30,64,175,0.45);border-radius:12px;padding:6px 10px;background:rgba(37,99,235,0.06)}
.mon-ai-holder summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:6px;color:#1d4ed8;font-size:12px;font-weight:600}
.mon-ai-holder summary::-webkit-details-marker{display:none}
.mon-ai-holder[open]{background:rgba(37,99,235,0.12)}
.mon-ai-holder div{margin-top:6px;font-size:12px;color:#1f2937;line-height:1.4}
.mon-sticky-wrap{position:sticky;bottom:0;margin-top:22px;z-index:20}
.mon-sticky{background:rgba(17,24,39,0.92);color:#fff;border-radius:14px;padding:10px 14px;display:flex;align-items:center;gap:12px;overflow:hidden}
.mon-chip-list{display:flex;gap:10px;overflow-x:auto;scrollbar-width:thin;padding-bottom:4px}
.mon-chip{flex:0 0 auto;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.mon-chip.alarm{background:#ef4444}
.mon-chip.warn{background:#f59e0b;color:#111827}
.mon-chip.ok{background:#10b981;color:#fff}
.mon-chevron{margin-left:auto;font-size:18px;cursor:pointer}
.mon-chevron a{color:#fff;text-decoration:none;display:inline-flex;align-items:center;padding:4px}
.mon-empty{font-size:12px;color:#9ca3af}
</style>
""", unsafe_allow_html=True)

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
        pwd = st.sidebar.text_input("Пароль", type="password", value="admin")
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

# MONITOR: модель данных и хелперы
Status = Literal["ok", "warn", "alarm"]


@dataclass
class Violation:
    label_ru: str
    prob: float
    ttf_hours: Optional[float] = None


@dataclass
class MotorCard:
    id: str
    shop: str
    equip_name: str
    status: Status
    violations: List[Violation]
    info_badge: bool = True
    updated_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz=None))


_CLASS_LABELS_RU = {
    "normal": "Норма",
    "BPFO": "Дефект наружного кольца",
    "BPFI": "Дефект внутреннего кольца",
    "BSF": "Дефект тел качения",
    "FTF": "Дефект сепаратора",
    "imbalance": "Дисбаланс",
    "misalignment": "Разцентровка",
}


def _advice_text(status: Status) -> str:
    if status == "alarm":
        return (
            "• Приоритизируйте останов и осмотр.\n"
            "• Проверьте ток и виброканалы, разницу фаз, нагрузку."
        )
    if status == "warn":
        return (
            "• Запланируйте диагностику в ближайшее окно.\n"
            "• Сверьте люфты и балансировку."
        )
    return "• Отклонений тревожного уровня не найдено."


def _violations_html(card: MotorCard) -> str:
    parts = []
    for viol in card.violations[:2]:
        pill_class = f"mon-pill {card.status if card.status in ('alarm', 'warn') else 'ok'}"
        parts.append(
            f"<span class='{pill_class}'>{viol.prob * 100:.0f}%&nbsp;{html.escape(viol.label_ru)}</span>"
        )
        if viol.ttf_hours and math.isfinite(viol.ttf_hours):
            parts.append(
                f"<div class='mon-ttf'>Отказ через {viol.ttf_hours:.0f} ч</div>"
            )
    if not parts:
        parts.append("<span class='mon-pill ok'>0%&nbsp;Нарушений нет</span>")
    return "".join(parts)


def generate_demo_monitor(seed: int = 42) -> List[MotorCard]:
    random.seed(seed)
    shops = ["Агрегат 1", "Агрегат 2", "Агрегат 3"]
    equips = [
        "Насос компрессора 101",
        "Вентилятор охлаждения 5",
        "Генератор линии 2",
        "Дымосос 4",
        "Компрессор подачи 7",
        "Помпа охлаждения 3",
        "Привод мешалки 8",
        "Вибропресс 6",
    ]
    statuses = ["ok", "warn", "alarm"]
    cards: List[MotorCard] = []
    count = random.randint(8, 12)
    for idx in range(count):
        status = random.choices(statuses, weights=[0.45, 0.35, 0.2])[0]
        equip = random.choice(equips)
        shop = random.choice(shops)
        card_id = f"U-{10 + idx}"
        viols: List[Violation] = []
        if status in ("warn", "alarm"):
            top_classes = random.sample(_LABELS_ALL[1:], k=2)
            for j, cls in enumerate(top_classes):
                prob = 0.55 + 0.4 * random.random() if j == 0 else 0.3 + 0.3 * random.random()
                viols.append(
                    Violation(
                        label_ru=_CLASS_LABELS_RU.get(cls, cls),
                        prob=min(prob, 0.99),
                        ttf_hours=(4.0 + random.random() * 12.0) if status == "alarm" and j == 0 else None,
                    )
                )
        cards.append(
            MotorCard(
                id=card_id,
                shop=shop,
                equip_name=equip,
                status=status,  # type: ignore[arg-type]
                violations=viols,
                info_badge=bool(random.getrandbits(1)),
                updated_at=pd.Timestamp.now(tz=None),
            )
        )
    return cards


@st.cache_resource(show_spinner=False)
def _load_predictor_cached():
    try:
        from inference_v2 import load_predictor

        return load_predictor()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _cached_file_summary(path: str, limit_seconds: float) -> Optional[dict]:
    predictor = _load_predictor_cached()
    if predictor is None:
        return None
    try:
        df, agg = predictor.predict_csv(
            path,
            fs_raw=25600,
            fs_out=3200,
            win_sec=1.0,
            overlap=0.0,
        )
    except Exception:
        return None

    if isinstance(df, pd.DataFrame):
        if "t1" in df.columns:
            df_small = df[df["t1"] <= limit_seconds]
        else:
            df_small = df.copy()
        if df_small.empty:
            df_small = df.head(3)
    else:
        df_small = pd.DataFrame(df)

    if df_small.empty:
        return None

    thr = float(agg.get("thr_bin", 0.5)) if isinstance(agg, dict) else 0.5
    p_def_series = pd.to_numeric(df_small.get("p_def", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    p_def_mean = float(p_def_series.mean()) if not p_def_series.empty else 0.0
    share = float((p_def_series >= thr).mean()) if not p_def_series.empty else 0.0

    status: Status
    if p_def_mean >= thr or share >= 0.2:
        status = "alarm"
    elif p_def_mean >= 0.5:
        status = "warn"
    else:
        status = "ok"

    if "p7" in df_small:
        rows = []
        for val in df_small["p7"]:
            try:
                rows.append(np.asarray(val, dtype=float))
            except Exception:
                continue
        arr = np.vstack(rows) if rows else np.array([])
    else:
        arr = np.array([])
    if arr.size:
        mean_p7 = arr.mean(axis=0)
    else:
        mean_p7 = np.zeros(len(_LABELS_ALL))
        mean_p7[0] = 1.0
    top_idx = np.argsort(mean_p7)[::-1][:2]
    violations = []
    for idx in top_idx:
        label = _LABELS_ALL[idx] if idx < len(_LABELS_ALL) else "normal"
        violations.append(
            {
                "label_ru": _CLASS_LABELS_RU.get(label, label),
                "prob": float(mean_p7[idx]),
            }
        )
    ttf_hours = None
    if isinstance(agg, dict) and agg.get("ttf_to_80_sec") not in (None, float("inf")):
        try:
            ttf_hours = float(agg["ttf_to_80_sec"]) / 3600.0
        except Exception:
            ttf_hours = None
    return {
        "status": status,
        "violations": violations,
        "ttf_hours": ttf_hours,
        "p_def_mean": p_def_mean,
        "share": share,
    }


def fetch_monitor_data(calc_live: bool, file_limit: int = 12, limit_seconds: float = 12.0) -> List[MotorCard]:
    if not calc_live:
        return generate_demo_monitor()

    files = sorted(glob.glob("data/raw/*.csv"))[: file_limit or 1]
    if not files:
        return generate_demo_monitor()

    cards: List[MotorCard] = []
    for idx, path in enumerate(files):
        summary = _cached_file_summary(path, limit_seconds)
        if summary is None:
            continue
        violations = [
            Violation(
                label_ru=v.get("label_ru", "Аномалия"),
                prob=float(v.get("prob", 0.0)),
                ttf_hours=summary.get("ttf_hours") if n == 0 else None,
            )
            for n, v in enumerate(summary.get("violations", [])[:2])
        ]
        equip_name = os.path.splitext(os.path.basename(path))[0]
        cards.append(
            MotorCard(
                id=f"CSV-{idx + 1}",
                shop=f"Цех {1 + (idx % 3)}",
                equip_name=equip_name,
                status=summary.get("status", "ok"),  # type: ignore[arg-type]
                violations=violations,
                info_badge=True,
                updated_at=pd.Timestamp.now(tz=None),
            )
        )

    if not cards:
        return generate_demo_monitor()
    return cards


def build_notifications_from_cards(cards: List[MotorCard]) -> pd.DataFrame:
    rows = []
    for card in cards:
        if card.status not in ("alarm", "warn"):
            continue
        msg = card.violations[0].label_ru if card.violations else "Аномалия"
        if card.violations and card.violations[0].ttf_hours:
            msg += f" · Отказ через {card.violations[0].ttf_hours:.0f} ч"
        rows.append(
            {
                "ts": card.updated_at,
                "severity": card.status,
                "shop": card.shop,
                "equip": card.equip_name,
                "message": msg,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ts", "severity", "shop", "equip", "message"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df.sort_values("ts")


def _render_notifications_bar(notifications: pd.DataFrame) -> None:
    if notifications.empty:
        st.markdown(
            """
            <div class="mon-sticky-wrap">
                <div class="mon-sticky"><div class="mon-empty">Активных уведомлений нет</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    chips = []
    for _, row in notifications.iterrows():
        ts = pd.to_datetime(row["ts"]).strftime("%d.%m.%Y %H:%M")
        sev = row.get("severity", "warn")
        shop = html.escape(str(row.get("shop", "Цех")))
        equip = html.escape(str(row.get("equip", "Оборудование")))
        message = html.escape(str(row.get("message", "Событие")))
        chips.append(
            f"<span class='mon-chip {sev}'>{ts} | {shop} {equip} | {message}</span>"
        )

    st.markdown(
        """
        <div class="mon-sticky-wrap">
            <div class="mon-sticky">
                <div class="mon-chip-list">{chips}</div>
                <div class="mon-chevron"><a href="#monitor-notifications-panel">▶</a></div>
            </div>
        </div>
        """.format(chips="".join(chips)),
        unsafe_allow_html=True,
    )

    st.markdown("<div id='monitor-notifications-panel'></div>", unsafe_allow_html=True)

    with st.expander("Полный список уведомлений", expanded=False):
        st.dataframe(
            notifications.sort_values("ts", ascending=False),
            use_container_width=True,
        )
        csv_bytes = notifications.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Скачать уведомления (CSV)",
            data=csv_bytes,
            file_name="notifications.csv",
            mime="text/csv",
        )


def render_monitor(cards: List[MotorCard], notifications: pd.DataFrame) -> None:
    st.title("ИИмпульс • Монитор диспетчера")
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Цех 1")
    with header_cols[1]:
        st.caption(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

    expand_block = st.toggle("Развернуть блок", value=True, key="monitor_expand")
    if not expand_block:
        _render_notifications_bar(notifications)
        return

    cards_html = ["<div class='mon-grid'>"]
    for card in cards:
        status_class = card.status if card.status in ("alarm", "warn") else "ok"
        badge_html = "<span class='mon-badge'>инфо по позиции</span>"
        viols = _violations_html(card)
        equip_html = html.escape(card.equip_name)
        shop_html = html.escape(card.shop)
        card_id_html = html.escape(card.id)
        advice_html = html.escape(_advice_text(card.status)).replace("\n", "<br>")
        card_html = "".join(
            [
                f"<div class='mon-card {status_class}'>",
                "<div class='mon-card-head'>",
                f"<span class='mon-card-title'>{card_id_html}</span>",
                badge_html,
                "</div>",
                f"<div class='mon-card-equip'>{equip_html}</div>",
                f"<div class='mon-card-shop'>{shop_html}</div>",
                f"<div class='mon-card-viol'>Нарушения:<br>{viols}</div>",
                "<div class='mon-card-caption'>ТС: под-пороговые сигналы контролируются</div>",
                f"<details class='mon-ai-holder'><summary>❓ ИИ-помощник</summary><div>{advice_html}</div></details>",
                "</div>",
            ]
        )
        cards_html.append(card_html)
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)

    _render_notifications_bar(notifications)


# --- PAGES --- скорректированы, для видео - часть страниц в проработке
page = st.sidebar.radio(
    "Разделы",
    [
        "ИИмпульс • Монитор диспетчера",
        "ИИмпульс • Сводка",
        "ИИмпульс • Прогноз поломок",
        "ИИ-предсказание (MVP)",
    ],
)

if page.startswith("ИИмпульс • Монитор диспет"):
    with st.sidebar:
        st.subheader("Монитор диспетчера")
        calc_live = st.checkbox(
            "Рассчитать по CSV (медленно)",
            value=False,
            help="Если включено — для CSV в data/raw/ запустим быструю оценку инференса.",
        )
        file_limit = st.number_input(
            "Лимит файлов",
            min_value=1,
            max_value=48,
            value=12,
            step=1,
            help="Ограничивает число CSV для расчёта, чтобы не блокировать интерфейс.",
            disabled=not calc_live,
        )
        autorefresh = st.checkbox(
            "Автообновление (30 с)",
            value=True,
            help="При включении страница обновляется каждые 30 секунд.",
        )
        st.caption("Без моделей или файлов используются демо-данные для карточек.")

    if autorefresh:
        st_autorefresh(interval=30000, key="monitor_refresh")

    cards = fetch_monitor_data(calc_live=calc_live, file_limit=int(file_limit))
    notifications = build_notifications_from_cards(cards)
    render_monitor(cards, notifications)

elif page.startswith("ИИмпульс • Сводка"):

    st.title("ИИмпульс • Сводка")
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
            is_warn = (df["severity"] >= warn_thr) | (df["p_fault"] >= (warn_thr / 100.0))

            alarms += int(is_alarm.sum())
            warns += int(is_warn.sum())

            # распределение источников дефектов (по окнам, где есть предупреждение)
            if "proba" in df.columns:
                probs = np.array(df.loc[is_warn, "proba"].tolist()) if is_warn.any() else np.empty((0, 6))
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
        c2.metric("Предупреждения (≥" + str(thresh_warn) + ")", "4")
        # «из 1С ТОИР» — заглушка
        repairs = st.session_state.get("repair_count", 3)
        c3.metric("Устройств в ремонте (1С ТОИР)", repairs)
        c4.metric("Двигателей на мониторинге", len(files))
    else:
        c1.metric("Алармы (severity≥80 / p≥0.8)", f"{summary['alarms']}")
        c2.metric("Предупреждения (≥" + str(thresh_warn) + ")", f"{summary['warns']}")
        # «из 1С ТОИР» — сейчас заглушка, позже сюда коннектор
        repairs = st.session_state.get("repair_count", 3)
        c3.metric("Устройств в ремонте (1С ТОИР)", repairs)
        c4.metric("Двигателей на мониторинге", len(files))

        # круговая диаграмма по источникам дефектов
        st.subheader("Распределение вероятных источников дефектов (усреднение по окнам с предупреждениями)")
        labels = ["bearing_outer", "bearing_inner", "rolling", "cage", "imbalance", "misalignment"]
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
    n = len(x);
    dur = n / fs

    # позиция фрагмента
    pos = st.slider("Позиция начала (сек)", 0.0, max(0.0, dur - show_sec), 0.0, 0.5)
    i0 = int(pos * fs);
    i1 = min(i0 + int(show_sec * fs), n)
    frag = x[i0:i1]

    # downsample для рисования
    max_points = 8000
    step = max(1, (i1 - i0) // max_points)
    t = np.arange(i0, i1, step) / fs
    frag_ds = frag[::step]

    fig = go.Figure()
    for i, name in enumerate(["A", "B", "C"]):
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
    st.title("🔮 ИИ-диагностика по CSV (MVP)")
    st.caption(
        "Загрузите CSV (3 колонки тока A,B,C) — получим вероятность дефекта, тип, тяжесть и прогноз времени до тревоги (TTF).")

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
        fs_out = st.select_slider("Частота обработки, Гц", [2000, 3200, 5120], value=3200)
        win_sec = st.select_slider("Окно, сек", [0.5, 1.0, 2.0], value=1.0)
        overlap = st.select_slider("Перекрытие", [0.0, 0.25, 0.5], value=0.0)
        st.divider()
        with st.expander("Параметры подшипника / RPM (опционально)", expanded=False):
            rpm = st.number_input("RPM", 500, 6000, 1770, step=10)
            Z = st.number_input("Z (число тел качения)", 3, 20, 9, step=1)
            dmm = st.number_input("d, мм (шарик/ролик)", value=7.94, step=0.01, format="%.2f")
            Dmm = st.number_input("D, мм (делительная)", value=38.5, step=0.1, format="%.1f")
            theta = st.number_input("Угол контакта, °", value=0, step=1)
        st.caption(f"Порог детектора из модели: **{predictor.info.get('bin_threshold', 0.5):.3f}**")

    # ===== ФАЙЛ =====
    st.subheader("Данные")

    # храним выбранный демо-путь между перерисовками
    st.session_state.setdefault("demo_path", None)

    c_up, c_demo = st.columns([4, 1])
    with c_up:
        up = st.file_uploader("Загрузите CSV с тремя колонками (A,B,C)", type=["csv"])
    with c_demo:
        demo_clicked = st.button("🎯 ДЕМО-ЗАГРУЗКА", help="Подгрузит data/raw/current_1.csv (или ближайший 1.csv)")

    data_bytes = None
    uploaded_name = None

    if up is not None:
        # обычная загрузка файла
        data_bytes = up.read()
        uploaded_name = up.name
        st.session_state["demo_path"] = None  # если был демо-режим — отключаем
    else:
        # если нажали кнопку или демо уже выбран ранее — берём демо-файл
        if demo_clicked or st.session_state.get("demo_path"):
            import os

            demo_path = st.session_state.get("demo_path")
            if not demo_path:
                # ищем 1.csv в типовых местах
                for p in ["data/raw/current_1.csv", "data/demo/current_1.csv", "data/current_1.csv", "current_1.csv"]:
                    if os.path.exists(p):
                        demo_path = p
                        st.session_state["demo_path"] = p
                        break
            if demo_path and os.path.exists(demo_path):
                with open(demo_path, "rb") as f:
                    data_bytes = f.read()
                uploaded_name = os.path.basename(demo_path)
                st.info(f"Используем демо-файл: **{uploaded_name}**")
            else:
                st.error("Демо-файл **1.csv** не найден. Положите его в `data/raw/` (или `data/demo/`, `data/`).")
                st.stop()

    # если ничего не выбрано — ждём файл/кнопку
    if data_bytes is None:
        st.stop()

    # ===== ИНФЕРЕНС =====
    try:
        with st.spinner("Считаем предсказания…"):
            df_pred, agg = predictor.predict_csv(
                io.BytesIO(data_bytes),
                fs_raw=25600, fs_out=fs_out,
                win_sec=win_sec, overlap=overlap,
                rpm=float(locals().get("rpm", 1770)),
                Z=int(locals().get("Z", 9)),
                dmm=float(locals().get("dmm", 7.94)),
                Dmm=float(locals().get("Dmm", 38.5)),
                theta=int(locals().get("theta", 0))
            )
    except Exception as e:
        st.error(f"Ошибка инференса: {e}")
        st.stop()

    # ===== РУС ИМЕНА КЛАССОВ =====
    ru7 = {
        "normal": "норма",
        "BPFO": "подш. наружное кольцо",
        "BPFI": "подш. внутреннее кольцо",
        "BSF": "дефект тел качения",
        "FTF": "сепаратор",
        "imbalance": "дисбаланс",
        "misalignment": "расцентровка",
    }
    ru3 = {
        "bearing": "подшипник",
        "rotor": "ротор",
        "stator": "статор",
        "other": "прочее",
        "none": "нет дефекта",
        "unknown": "—",
    }
    label7 = predictor.labels7 if hasattr(predictor, "labels7") else list(ru7.keys())

    # ===== KPI / ВЕРДИКТ =====
    thr = agg["thr_bin"]
    verdict_defect = (agg["p_def_mean"] >= thr) or (agg["p_def_share_over_thr"] >= 0.2)
    verdict_text = "ДЕФЕКТ" if verdict_defect else "Норма"
    emoji = "🚨" if verdict_defect else "✅"

    # Баннер (крупный монитор)
    if verdict_defect:
        st.markdown(f"""
        <div class="big-banner danger">
          <div style="font-size:28px;line-height:1.0">🔴</div>
          <div>
            <p class="title">{emoji} ВНИМАНИЕ! ОБНАРУЖЕН ДЕФЕКТ</p>
            <p class="subtitle">Система выявила повышенный риск по загруженному файлу. ПРОВЕРЬТЕ оборудование и оцените тренды на графиках ниже.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="big-banner ok">
          <div style="font-size:28px;line-height:1.0">🟢</div>
          <div>
            <p class="title">{emoji} Норма — дефектов не обнаружено</p>
            <p class="subtitle">Вероятность дефекта ниже порога. Для уверенности просмотрите тренды и таблицу «окна риска».</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Чипы классов файла
    file_class7_ru = ru7.get(agg["file_class7"], agg["file_class7"])
    file_class3_ru = ru3.get(agg["file_class3"], agg["file_class3"])
    st.markdown(
        f'<span class="badge">Класс файла (7): <b>{file_class7_ru}</b></span>'
        f'<span class="badge">Класс файла (3): <b>{file_class3_ru}</b></span>',
        unsafe_allow_html=True
    )

    # Сетка KPI (упростили: без «Окон выше порога», показываем только максимум тяжести)
    p_mean = agg['p_def_mean']
    ttf = "∞" if np.isinf(agg["ttf_to_80_sec"]) else f"{agg['ttf_to_80_sec']:.0f} с"
    sev_max = agg['severity_max']

    kpi_html = f"""
        <div class="kpi-grid">
          <div class="kpi"><div class="label">Вердикт</div>
               <div class="value">{'🔴 ДЕФЕКТ' if verdict_defect else '🟢 НОРМА'}</div></div>
          <div class="kpi"><div class="label">Средняя вероятность дефекта</div>
               <div class="value">{p_mean:.2f}</div></div>
          <div class="kpi"><div class="label">Время до отказа</div>
               <div class="value">{ttf}</div></div>
        </div>
        <div class="kpi-grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">
          <div class="kpi"><div class="label">Класс дефекта</div>
               <div class="value">{file_class7_ru}</div></div>
          <div class="kpi"><div class="label">Причина дефекта</div>
               <div class="value">{file_class3_ru}</div></div>
        </div>
        """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Пояснение к «Тяжести»
    # st.caption("«Тяжесть (Severity)» — шкала 0–100: 0–49 — норма, 50–79 — предупреждение, ≥80 — тревога. "
    # "На карточке показан максимум по файлу; подробная кривая — во вкладке «Графики».")
    st.divider()

    # ===== ВКЛАДКИ: Графики / Таблица / Распределение =====
    tab_g, tab_t, tab_p = st.tabs(["Графики", "Окна риска", "Распределение классов"])

    with tab_g:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.subheader("Вероятность дефекта по времени")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_pred["t1"], y=df_pred["p_def"], mode="lines", name="p(defect)"))
            fig1.add_hline(y=thr, line_dash="dot", annotation_text="порог", annotation_position="top left")
            fig1.update_layout(height=300, xaxis_title="Время, с", yaxis_title="Вероятность")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.subheader("Тяжесть (0–100) по времени")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_pred["t1"], y=df_pred["severity"], mode="lines", name="severity"))
            fig2.add_hline(y=50, line_dash="dot", annotation_text="warning")
            fig2.add_hline(y=80, line_dash="dot", annotation_text="alarm")
            fig2.update_layout(height=300, xaxis_title="Время, с", yaxis_title="Тяжесть")
            st.plotly_chart(fig2, use_container_width=True)

    with tab_t:
        st.subheader("Топ-25 окон с наибольшим риском")
        top = df_pred.sort_values(["p_def", "severity"], ascending=False).head(25)[
            ["t0", "t1", "p_def", "y_pred", "severity"]]
        st.dataframe(top, use_container_width=True, height=360)
        st.download_button(
            "⬇️ Скачать отчёт по окнам (CSV)",
            df_pred.to_csv(index=False).encode("utf-8"),
            file_name=f"report_{uploaded_name or 'demo.csv'}",
            mime="text/csv"
        )

    with tab_p:
        st.subheader("Распределение типов дефектов (усреднение по файлу)")
        labels7 = label7
        p7 = np.array(agg["file_p7"])
        # русские подписи при наличии
        labels7_ru = [ru7.get(x, x) for x in labels7]
        df_p7 = pd.DataFrame({"Класс": labels7_ru, "Вероятность": p7})
        fig_p7 = px.bar(df_p7, x="Класс", y="Вероятность", text="Вероятность")
        fig_p7.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_p7.update_layout(height=320, yaxis_range=[0, 1], margin=dict(t=20))
        st.plotly_chart(fig_p7, use_container_width=True)

elif ("Прогноз поломок" in page):
    st.title("ИИмпульс • Прогноз поломок")
    st.caption("Загрузи журнал или работай на демо-данных. Фильтры — слева.")

    import io, math, re, random
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    # ---- мини-стили для KPI/баннеров ----
    st.markdown("""
    <style>
      .kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:8px 0 16px}
      .kpi{padding:14px 16px;border-radius:12px;background:#F8FAFC;border:1px solid #EFF2F6}
      .kpi .label{font-size:12px;color:#64748B;text-transform:uppercase;letter-spacing:.02em}
      .kpi .value{font-size:24px;font-weight:800;margin-top:4px}
      .kpi .sub{font-size:12px;color:#6B7280;margin-top:2px}
      .banner-money{padding:16px 18px;border-radius:14px;border:1px solid #FECACA;background:#FEE2E2;color:#7F1D1D;margin:4px 0 12px}
      .badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#F3F4F6;font-weight:600;font-size:13px;margin-right:8px}
      @media (max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    </style>
    """, unsafe_allow_html=True)

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
        if isinstance(x, str): x = x.strip().lower()
        return x in (True, 1, "1", "true", "да", "y", "yes", "ок", "истина")


    def _fmt_money(v, cur="₽"):
        try:
            return f"{int(round(float(v))):,}".replace(",", " ") + f" {cur}"
        except Exception:
            return "—"


    @st.cache_data(show_spinner=False)
    def load_uploaded(file):
        if file is None: return None
        try:
            return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file, engine="openpyxl")
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
            t0 = (datetime(2000, 1, 1, 8, 0) + timedelta(
                minutes=int(np.random.rand() * 600))).time() if np.random.rand() < 0.8 else None
            sh = random.choice(shops);
            tp = random.choice(types);
            eq = random.choice(eqs)
            kind = np.random.choice(kinds, p=[0.36, 0.32, 0.1, 0.12, 0.1])
            crit = np.random.choice(crits, p=[0.25, 0.5, 0.25])
            stt = np.random.choice(statuses, p=[0.25, 0.15, 0.60])
            created_dt = datetime.combine(d0, t0 or datetime.min.time()) + timedelta(hours=np.random.uniform(0, 6))
            plan_dt = created_dt + timedelta(hours=np.random.uniform(4, 72))
            closed = (stt == "Устранена") and (np.random.rand() < 0.9)
            fix_dt = created_dt + timedelta(hours=np.random.uniform(2, 48)) if closed else pd.NaT
            downtime = float(np.round(np.random.uniform(0.5, 24), 2)) if closed else np.nan
            cost = float(np.round(np.exp(np.random.normal(6, 0.9)) / 100, 2)) if closed else np.nan
            rows.append({
                "ID записи": i + 1, "Дата обнаружения": d0, "Время обнаружения": t0, "Цех / участок": sh,
                "Оборудование / двигатель": eq, "Инвентарный № / серийный №": f"INV-{10000 + i}",
                "Тип оборудования": tp, "Тип поломки": kind, "Описание неисправности": f"Симптомы {kind} на {eq}",
                "Критичность": crit, "Создана заявка": "Да" if np.random.rand() < 0.85 else "Нет",
                "Номер заявки": f"REQ-{200000 + i}" if np.random.rand() < 0.85 else "",
                "Ответственный за устранение": random.choice(resp),
                "Дата/время создания заявки": created_dt, "Срок устранения (план)": plan_dt,
                "Дата фактического устранения": fix_dt, "Время простоя": downtime, "Статус": stt,
                "Причина поломки": np.random.choice(reasons, p=[0.35, 0.15, 0.15, 0.1, 0.15, 0.10]),
                "Принятые меры": "Диагностика/ремонт/замена", "Исполнитель ремонта": random.choice(execs),
                "Стоимость ремонта / запчастей": cost, "Комментарии": ""
            })
        return pd.DataFrame(rows)


    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing: st.error("Не хватает колонок: " + ", ".join(missing))
        df = df.copy()
        df["Дата обнаружения"] = pd.to_datetime(df["Дата обнаружения"], errors="coerce").dt.date
        if "Время обнаружения" in df:
            df["Время обнаружения"] = pd.to_datetime(df["Время обнаружения"], errors="coerce").dt.time
        df["Дата/время создания заявки"] = pd.to_datetime(df["Дата/время создания заявки"], errors="coerce")
        df["Срок устранения (план)"] = pd.to_datetime(df["Срок устранения (план)"], errors="coerce")
        df["Дата фактического устранения"] = pd.to_datetime(df["Дата фактического устранения"], errors="coerce")
        df["Создана заявка"] = df["Создана заявка"].map(_to_bool)
        for col in ["Время простоя", "Стоимость ремонта / запчастей"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        ts_detect = pd.to_datetime(df["Дата обнаружения"], errors="coerce")
        if "Время обнаружения" in df and df["Время обнаружения"].notna().any():
            ts_detect = pd.to_datetime(df["Дата обнаружения"].astype(str) + " " + df["Время обнаружения"].astype(str),
                                       errors="coerce")
        df["__ts_detect"] = ts_detect
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
        mtbf_vals = []
        for eq, g in df.sort_values("__ts_detect").groupby("Оборудование / двигатель"):
            times = pd.to_datetime(g["__ts_detect"]).values
            if len(times) >= 2:
                diffs = np.diff(times).astype("timedelta64[h]").astype(float)
                if len(diffs) > 0: mtbf_vals.append(diffs.mean())
        mtbf = np.mean(mtbf_vals) if mtbf_vals else np.nan
        c3.metric("MTBF, ч", "—" if (not mtbf_vals or np.isnan(mtbf)) else f"{mtbf:.0f}")
        closed = df[df["Дата фактического устранения"].notna()].copy()
        if len(closed):
            ontime = (closed["Дата фактического устранения"] <= closed["Срок устранения (план)"]).mean() * 100
            c4.metric("Доля «в срок», %", f"{ontime:.0f}")
        else:
            c4.metric("Доля «в срок», %", "—")
        st.caption(
            "MTTR — среднее по закрытым инцидентам. MTBF — средний интервал между отказами (по оборудованию с ≥2 событиями).")


    # ---------- загрузка/демо ----------
    with st.sidebar:
        st.subheader("Источник данных")
        upl = st.file_uploader("Загрузить журнал (CSV/XLSX)", type=["csv", "xlsx"], key="toir_upload")
    df_raw = load_uploaded(upl)
    if df_raw is None: df_raw = make_demo()
    df = normalize_df(df_raw)

    # ---------- фильтры + KPI ----------
    dff = apply_filters(df)
    kpi_cards(dff)

    # ---------- вкладки ----------
    tab_money, tab0, tab1, tab2, tab3, tab_fin, tab5 = st.tabs(
        ["Деньги (C-Level)", "Общее", "Оборудование", "Типы поломок", "Сроки и эффективность", "Финансы (детально)",
         "Прогноз"]
    )

    # ======== ВКЛАДКА ДЛЯ ТОП-МЕНЕДЖМЕНТА ========
    with tab_money:
        st.subheader("Финансовая оценка потерь")
        c_rate, c_cur = st.columns([2, 1])
        with c_rate:
            cost_per_hour = st.number_input("Стоимость часа простоя, ₽/час", min_value=0, value=120_000, step=5_000,
                                            help="Оценка потерь выручки/маржи в час простоя линии/агрегата.")
        with c_cur:
            currency = st.selectbox("Валюта отображения", ["₽", "₸", "₴", "$", "€"], index=0)

        # подготовка денежных колонок
        money = dff.copy()
        money["__downtime_h"] = pd.to_numeric(money["Время простоя"], errors="coerce").fillna(0.0)
        money["__repair_cost"] = pd.to_numeric(money["Стоимость ремонта / запчастей"], errors="coerce").fillna(0.0)
        money["__downtime_cost"] = money["__downtime_h"] * float(cost_per_hour)
        money["__loss_total"] = money["__downtime_cost"] + money["__repair_cost"]

        total_downtime_h = float(money["__downtime_h"].sum())
        downtime_cost = float(money["__downtime_cost"].sum())
        repair_cost = float(money["__repair_cost"].sum())
        total_loss = float(money["__loss_total"].sum())

        # период для окупаемости
        if len(money):
            period_days = (pd.to_datetime(money["__ts_detect"]).max() - pd.to_datetime(
                money["__ts_detect"]).min()).days + 1
        else:
            period_days = 30
        months = max(1.0, period_days / 30.44)

        # большой баннер
        st.markdown(
            f'<div class="banner-money">💸 <b>Потери за период:</b> {_fmt_money(total_loss, currency)} '
            f'• простой: {_fmt_money(downtime_cost, currency)} • ремонт/запчасти: {_fmt_money(repair_cost, currency)}</div>',
            unsafe_allow_html=True
        )

        # KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""<div class="kpi"><div class="label">Суммарный простой</div>
                        <div class="value">{total_downtime_h:.0f} ч</div><div class="sub">средний на инцидент — {money["__downtime_h"].replace(0, np.nan).mean():.1f} ч</div></div>""",
                    unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi"><div class="label">Потери от простоя</div>
                        <div class="value">{_fmt_money(downtime_cost, currency)}</div><div class="sub">{int(round(cost_per_hour)):,} ₽/час</div></div>""".replace(
            ",", " "), unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi"><div class="label">Расходы на ремонт/запчасти</div>
                        <div class="value">{_fmt_money(repair_cost, currency)}</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="kpi"><div class="label">Итого прямые потери</div>
                        <div class="value">{_fmt_money(total_loss, currency)}</div><div class="sub">период ≈ {months:.1f} мес.</div></div>""",
                    unsafe_allow_html=True)

        st.divider()

        # Сценарии экономии и ROI
        st.subheader("Сценарий экономии и ROI")
        s1, s2, s3, s4 = st.columns([1, 1, 1, 1])
        with s1:
            mttr_red = st.slider("Снижение MTTR, %", 0, 30, 10, help="Влияние только на потери простоя.")
        with s2:
            fail_red = st.slider("Снижение количества отказов, %", 0, 30, 15, help="Влияет на простой и ремонт.")
        with s3:
            proj_cost = st.number_input("Стоимость проекта, ₽", min_value=0, value=5_000_000, step=250_000)
        with s4:
            st.caption("Пояснение: новая базовая линия считается для того же периода данных.")

        mttr_k = 1 - mttr_red / 100.0
        fail_k = 1 - fail_red / 100.0
        new_downtime_cost = downtime_cost * mttr_k * fail_k
        new_repair_cost = repair_cost * fail_k
        new_total_loss = new_downtime_cost + new_repair_cost
        savings_abs = total_loss - new_total_loss
        monthly_saving = savings_abs / months if months else savings_abs
        payback_months = (proj_cost / monthly_saving) if monthly_saving > 0 else np.inf
        roi = ((savings_abs - proj_cost) / proj_cost * 100.0) if proj_cost > 0 else np.nan

        cA, cB, cC, cD = st.columns(4)
        cA.markdown(f"""<div class="kpi"><div class="label">Экономия по сценарию</div>
                        <div class="value">{_fmt_money(savings_abs, currency)}</div><div class="sub">в мес.: {_fmt_money(monthly_saving, currency)}</div></div>""",
                    unsafe_allow_html=True)
        cB.markdown(f"""<div class="kpi"><div class="label">Новые потери</div>
                        <div class="value">{_fmt_money(new_total_loss, currency)}</div><div class="sub">простой {_fmt_money(new_downtime_cost, currency)} • ремонт {_fmt_money(new_repair_cost, currency)}</div></div>""",
                    unsafe_allow_html=True)
        cC.markdown(f"""<div class="kpi"><div class="label">ROI</div>
                        <div class="value">{'—' if np.isnan(roi) else f'{roi:.0f}%'} </div><div class="sub">при стоимости проекта {_fmt_money(proj_cost, currency)}</div></div>""",
                    unsafe_allow_html=True)
        cD.markdown(f"""<div class="kpi"><div class="label">Окупаемость</div>
                        <div class="value">{'∞' if np.isinf(payback_months) else f'{payback_months:.1f} мес.'}</div><div class="sub">расчёт по текущему периоду</div></div>""",
                    unsafe_allow_html=True)

        st.divider()

        # Pareto по оборудованию
        st.subheader("Pareto 80/20: где теряем деньги (оборудование)")
        loss_by_eq = money.groupby("Оборудование / двигатель")["__loss_total"].sum().sort_values(ascending=False)
        top_eq = loss_by_eq.head(15)
        if len(top_eq):
            cum = (top_eq.cumsum() / top_eq.sum() * 100).round(1)
            fig = go.Figure()
            fig.add_bar(x=top_eq.index, y=top_eq.values, name="Потери")
            fig.add_scatter(x=top_eq.index, y=cum.values, name="Накопленный %", yaxis="y2", mode="lines+markers")
            fig.update_layout(height=360, xaxis_title="Оборудование", yaxis_title=f"Потери, {currency}",
                              yaxis2=dict(title="% к итогу", overlaying="y", side="right", range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для расчёта Pareto.")

        # Водопад: из чего складываются потери
        st.subheader("Вклад факторов в потери (водопад)")
        figw = go.Figure(go.Waterfall(
            name="Потери",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Простой", "Ремонт/запчасти", "Итого"],
            text=[_fmt_money(downtime_cost, currency), _fmt_money(repair_cost, currency),
                  _fmt_money(total_loss, currency)],
            y=[downtime_cost, repair_cost, 0]
        ))
        figw.update_layout(height=320, showlegend=False, yaxis_title=f"{currency}")
        st.plotly_chart(figw, use_container_width=True)

        # Тепловая карта потерь по цехам/типам
        st.subheader("Потери по цехам × типам поломок")
        hm = money.pivot_table(values="__loss_total", index="Тип поломки", columns="Цех / участок", aggfunc="sum",
                               fill_value=0.0)
        if hm.size:
            fig_hm = px.imshow(hm, text_auto=".0f", aspect="auto", labels=dict(color=f"Потери, {currency}"))
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.caption("Недостаточно данных для тепловой карты.")

        # Топ-инциденты по деньгам
        st.subheader("ТОП-20 инцидентов по потерям")
        top_inc = money.copy()
        top_inc = top_inc.sort_values("__loss_total", ascending=False).head(20)[[
            "__ts_detect", "Цех / участок", "Оборудование / двигатель", "Тип поломки", "Критичность",
            "Время простоя", "Стоимость ремонта / запчастей", "__loss_total"
        ]].rename(columns={
            "__ts_detect": "Дата/время", "Время простоя": "Простой, ч", "Стоимость ремонта / запчастей": "Ремонт, ₽",
            "__loss_total": "Потери, ₽"
        })
        st.dataframe(top_inc, use_container_width=True, height=380)

        # Экспорт финансового отчёта
        try:
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as wr:
                money.to_excel(wr, index=False, sheet_name="Инциденты_с_оценкой")
                loss_by_eq.rename("Потери").reset_index().to_excel(wr, index=False, sheet_name="Потери_по_оборудованию")
                hm.reset_index().to_excel(wr, index=False, sheet_name="Потери_цех×тип")
                pd.DataFrame({
                    "Метрика": ["Потери от простоя", "Ремонт/запчасти", "Итого потерь", "Экономия по сценарию",
                                "Новые потери",
                                "ROI, %", "Окупаемость, мес.", "Стоимость часа, ₽/ч", "Период, мес."],
                    "Значение": [downtime_cost, repair_cost, total_loss, savings_abs, new_total_loss,
                                 None if np.isnan(roi) else roi, None if np.isinf(payback_months) else payback_months,
                                 cost_per_hour, months]
                }).to_excel(wr, index=False, sheet_name="Итоги")
            st.download_button("⬇️ Скачать финансовый отчёт (XLSX)", bio.getvalue(),
                               file_name="toir_finance_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.caption("Для XLSX установите пакет openpyxl.")

    # ======== Остальные вкладки (как были) ========
    with tab0:
        fr = st.radio("Шаг времени", ["День", "Неделя", "Месяц"], horizontal=True)
        code = {"День": "D", "Неделя": "W", "Месяц": "M"}[fr]
        ts = dff.set_index(pd.to_datetime(dff["__ts_detect"])).sort_index()
        if len(ts):
            trend = ts["ID записи"].resample(code).count().rename("count").to_frame()
            fig = px.line(trend, y="count", markers=True)
            fig.update_layout(height=320, xaxis_title="Дата", yaxis_title="Кол-во")
            st.plotly_chart(fig, use_container_width=True)
            gb = ts.groupby([pd.Grouper(freq=code), "Критичность"])["ID записи"].count().reset_index()
            fig2 = px.bar(gb, x="__ts_detect", y="ID записи", color="Критичность", barmode="stack")
            fig2.update_layout(height=320, xaxis_title="Дата", yaxis_title="Кол-во")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Нет данных для выбранных фильтров.")
        st.subheader("Последние инциденты")
        st.dataframe(dff.sort_values("__ts_detect", ascending=False).head(20), use_container_width=True)

    with tab1:
        c = dff["Оборудование / двигатель"].value_counts().head(5).reset_index()
        c.columns = ["Оборудование / двигатель", "Кол-во"]
        fig = px.bar(c, x="Кол-во", y="Оборудование / двигатель", orientation="h")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        pv = dff.pivot_table(index="Тип оборудования", columns="Цех / участок", values="ID записи", aggfunc="count",
                             fill_value=0)
        fig2 = px.imshow(pv, text_auto=True, aspect="auto")
        st.plotly_chart(fig2, use_container_width=True)
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
            g = closed.dropna(subset=["Дата/время создания заявки", "Дата фактического устранения"]).copy()
            if len(g):
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

    with tab_fin:
        cost_by_eq = dff.groupby("Оборудование / двигатель")["Стоимость ремонта / запчастей"].sum().sort_values(
            ascending=False).head(10)
        fig = px.bar(cost_by_eq.reset_index(), x="Оборудование / двигатель", y="Стоимость ремонта / запчастей")
        fig.update_layout(height=320, yaxis_title="Сумма, у.е.")
        st.plotly_chart(fig, use_container_width=True)
        cost_by_type = dff.groupby("Тип поломки")["Стоимость ремонта / запчастей"].sum().reset_index()
        fig2 = px.bar(cost_by_type, x="Тип поломки", y="Стоимость ремонта / запчастей")
        st.plotly_chart(fig2, use_container_width=True)
        agg_fin = dff.groupby("Тип поломки").agg(
            Стоимость=("Стоимость ремонта / запчастей", "sum"),
            Простой=("Время простоя", "sum"),
            Количество=("ID записи", "count")
        ).reset_index()
        fig3 = px.scatter(agg_fin, x="Стоимость", y="Простой", size="Количество", color="Тип поломки",
                          hover_name="Тип поломки")
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        eq_sel = st.selectbox("Оборудование для прогноза", sorted(dff["Оборудование / двигатель"].dropna().unique()))
        sub = dff[dff["Оборудование / двигатель"] == eq_sel].copy()
        sub["__ts_detect"] = pd.to_datetime(sub["__ts_detect"])
        sub = sub.dropna(subset=["__ts_detect"])
        ts = sub.set_index("__ts_detect").sort_index()
        if len(ts):
            weekly = ts["ID записи"].resample("W").count().rename("count").to_frame()
            weekly["sma"] = weekly["count"].rolling(3, min_periods=1).mean()
            last = weekly["sma"].iloc[-1]
            fut = pd.DataFrame({"sma": [last] * 4},
                               index=pd.date_range(weekly.index.max() + pd.Timedelta(days=7), periods=4, freq="W"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly["count"], mode="lines+markers", name="Факт/неделя"))
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly["sma"], mode="lines", name="SMA(3)"))
            fig.add_trace(go.Scatter(x=fut.index, y=fut["sma"], mode="lines", name="Прогноз", line=dict(dash="dash")))
            fig.update_layout(height=320, xaxis_title="Неделя", yaxis_title="Кол-во")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных по выбранному оборудованию.")

    st.divider()
    st.caption(
        "Экспорт всей текущей выборки — во вкладке «Деньги (C-Level)» и «Общее». Для PNG-экспорта графиков установите пакет `kaleido`.")

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
        up.seek(0);
        data_bytes = up.read()
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
        cls = ["bearing_outer", "bearing_inner", "rolling", "cage", "imbalance", "misalignment"]
        avg = np.array(df['proba'].tolist()).mean(axis=0)
        st.bar_chart(pd.DataFrame({"p": avg}, index=cls))

        st.download_button("Скачать отчёт (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="report.csv")