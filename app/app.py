import datetime
from datetime import timedelta
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_prediction_model.pkl"
DATA_PATH = BASE_DIR / "traffic.csv"

FEATURES = ["hour", "dayofweek", "month", "year", "dayofyear", "dayofmonth", "weekofyear"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PAGES = ["Dashboard", "Predict traffic", "Explore data"]

# ============================================================
# THEME
# ============================================================
C = {
    "bg": "#0F1317",
    "surface": "#171C22",
    "border": "#252C34",
    "text": "#E8EAED",
    "muted": "#8B95A1",
    "amber": "#F5A524",
    "green": "#3DD68C",
    "red": "#FF6B5B",
}

LEVEL_COLORS = {"Light": C["green"], "Moderate": C["amber"], "Heavy": C["red"]}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1, h2, h3, .hero-title, .big-number {{ font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; }}

.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
.brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem; }}
.brand-mark {{ width:34px; height:34px; border-radius:9px; background:{C['amber']}; display:grid; place-items:center; font-size:1.1rem; }}
.brand-name {{ font-family:'Bricolage Grotesque'; font-weight:700; font-size:1.15rem; }}
.side-note {{ color:{C['muted']}; font-size:.85rem; line-height:1.5; }}

/* Hero */
.hero {{
  position:relative; overflow:hidden; padding:2.2rem 2.2rem 2rem; border-radius:16px;
  background:{C['surface']}; border:1px solid {C['border']}; margin-bottom:1.5rem;
}}
.hero::after {{ /* road centre-line */
  content:""; position:absolute; left:0; right:0; bottom:0; height:6px;
  background:repeating-linear-gradient(90deg, {C['amber']} 0 36px, transparent 36px 64px);
}}
.hero-title {{ font-size:2.4rem; font-weight:700; margin:0 0 .4rem; line-height:1.1; }}
.hero-sub {{ color:{C['muted']}; max-width:56ch; font-size:1.02rem; line-height:1.55; margin:0; }}

/* Metrics */
[data-testid="stMetric"] {{
  background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:1rem 1.1rem;
}}
[data-testid="stMetricLabel"] p {{ color:{C['muted']}; font-size:.85rem; }}
[data-testid="stMetricValue"] {{ font-family:'Bricolage Grotesque'; font-weight:700; }}

/* Result card */
.result {{
  background:{C['surface']}; border:1px solid {C['border']}; border-radius:16px; padding:1.6rem 1.8rem; margin:1rem 0 1.4rem;
}}
.result-label {{ color:{C['muted']}; font-size:.9rem; margin-bottom:.2rem; }}
.big-number {{ font-size:3.6rem; font-weight:700; line-height:1; }}
.pill {{
  display:inline-block; padding:.25rem .8rem; border-radius:999px; font-size:.85rem; font-weight:500;
  margin-left:.8rem; vertical-align:middle;
}}
.compare {{ color:{C['muted']}; margin-top:.8rem; font-size:.95rem; }}
.meter {{ height:8px; border-radius:99px; background:{C['border']}; margin-top:1.1rem; overflow:hidden; }}
.meter > div {{ height:100%; border-radius:99px; }}

/* Feature list */
.feat {{ padding:1rem 0; border-top:1px solid {C['border']}; }}
.feat b {{ font-family:'Bricolage Grotesque'; font-size:1.05rem; }}
.feat p {{ color:{C['muted']}; margin:.25rem 0 0; font-size:.95rem; }}

/* Buttons & tabs */
.stButton > button {{ border-radius:10px; font-weight:500; padding:.6rem 1rem; }}
.stButton > button[kind="primary"] {{ color:#14171C; }}
button[data-baseweb="tab"] {{ font-size:.98rem; }}
:focus-visible {{ outline:2px solid {C['amber']} !important; outline-offset:2px; }}
@media (max-width:640px) {{ .hero-title {{ font-size:1.8rem; }} .big-number {{ font-size:2.6rem; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Matplotlib theme to match the app
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "axes.edgecolor": C["border"], "axes.labelcolor": C["muted"],
    "axes.titlecolor": C["text"], "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "grid.color": C["border"], "grid.alpha": 0.8, "text.color": C["text"],
    "legend.facecolor": C["surface"], "legend.edgecolor": C["border"],
    "font.size": 10,
})


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# DATA / MODEL
# ============================================================
def create_features(df):
    df = df.copy()
    dt = df["DateTime"].dt
    df["hour"] = dt.hour
    df["dayofweek"] = dt.dayofweek
    df["month"] = dt.month
    df["year"] = dt.year
    df["dayofyear"] = dt.dayofyear
    df["dayofmonth"] = dt.day
    df["weekofyear"] = dt.isocalendar().week.astype(int)
    return df


@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def junction_frame(junction):
    df = load_data()
    return create_features(df[df["Junction"] == junction]) if df is not None else None


def predict_batch(model, datetimes):
    frame = create_features(pd.DataFrame({"DateTime": pd.to_datetime(datetimes)}))
    return np.asarray(model.predict(frame[FEATURES]), dtype=float)


def traffic_level(value, ref):
    low, high = ref.quantile([0.33, 0.66])
    if value < low:
        return "Light"
    return "Moderate" if value < high else "Heavy"


def missing_file_error(label, path):
    st.error(f"{label} not found. Place it next to `app.py`:\n\n`{path}`")


# ============================================================
# NAVIGATION
# ============================================================
def go_to(page):
    st.session_state["page"] = page


def sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">🚦</div>'
            '<div class="brand-name">Traffic Intelligence</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="side-note">Forecast and explore junction traffic.</p>', unsafe_allow_html=True)
        st.write("")
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.write("")
        st.divider()
        st.markdown("**Model**")
        st.markdown(
            '<p class="side-note">XGBoost regression on time features: hour, weekday, month, '
            "year, day of year, day of month and week of year.</p>",
            unsafe_allow_html=True,
        )
    return page


# ============================================================
# DASHBOARD
# ============================================================
def show_home_page():
    st.markdown(
        '<div class="hero"><h1 class="hero-title">Know how busy the road will be.</h1>'
        '<p class="hero-sub">Pick a date and time to get a vehicle-count forecast, or dig into '
        "the historical patterns behind it.</p></div>",
        unsafe_allow_html=True,
    )

    j1 = junction_frame(1)
    if j1 is None:
        missing_file_error("Traffic dataset", DATA_PATH)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(j1):,}")
    c2.metric("Average vehicles / hour", f"{j1['Vehicles'].mean():,.0f}")
    c3.metric("Peak vehicles / hour", f"{j1['Vehicles'].max():,.0f}")
    c4.metric("Busiest hour", f"{j1.groupby('hour')['Vehicles'].mean().idxmax():02d}:00")

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("A typical week")
        grid = j1.pivot_table(index="dayofweek", columns="hour", values="Vehicles", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        sns.heatmap(grid, cmap="YlOrBr", ax=ax, cbar_kws={"label": "Avg vehicles", "shrink": 0.85},
                    yticklabels=DAY_NAMES, linewidths=0, rasterized=True)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
        show_fig(fig)

    with right:
        st.subheader("What you can do")
        st.markdown(
            '<div class="feat"><b>Forecast a moment</b><p>Get a vehicle count, a congestion level '
            "and how it compares with a usual hour.</p></div>"
            '<div class="feat"><b>Read the patterns</b><p>See how traffic shifts by hour, weekday '
            "and month.</p></div>"
            '<div class="feat"><b>Inspect the data</b><p>Filter any junction and date range, then '
            "download the records.</p></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    b1, b2 = st.columns(2)
    b1.button("Predict traffic", type="primary", use_container_width=True,
              on_click=go_to, args=("Predict traffic",))
    b2.button("Explore data", use_container_width=True, on_click=go_to, args=("Explore data",))


# ============================================================
# PREDICT
# ============================================================
def show_predict_page():
    st.title("Predict traffic")
    st.caption("Choose a date and time. Junction 1.")

    model = load_model()
    j1 = junction_frame(1)
    if model is None:
        missing_file_error("Model", MODEL_PATH)
        return
    if j1 is None:
        missing_file_error("Traffic dataset", DATA_PATH)
        return

    min_date = j1["DateTime"].min().date()
    max_hist = j1["DateTime"].max().date()
    max_allowed = max_hist + timedelta(days=365)

    col1, col2 = st.columns(2)
    with col1:
        pred_date = st.date_input(
            "Date", value=max_hist, min_value=min_date, max_value=max_allowed,
            help=f"Historical data runs to {max_hist:%d %b %Y}. Later dates are forecasts.",
        )
    with col2:
        pred_time = st.time_input("Time", value=datetime.time(12, 0), step=3600)

    is_future = pred_date > max_hist
    if is_future:
        st.info("This date is after the historical data, so the result is a pure forecast.")

    if not st.button("Predict traffic", type="primary", use_container_width=True):
        st.caption("Press Predict traffic to see the result.")
        return

    with st.spinner("Estimating traffic..."):
        when = datetime.datetime.combine(pred_date, pred_time)
        value = max(float(predict_batch(model, [when])[0]), 0.0)
        hours = [datetime.datetime.combine(pred_date, datetime.time(h)) for h in range(24)]
        curve = np.clip(predict_batch(model, hours), 0, None)

    level = traffic_level(value, j1["Vehicles"])
    color = LEVEL_COLORS[level]
    typical = j1.loc[(j1["hour"] == pred_time.hour) & (j1["dayofweek"] == pred_date.weekday()), "Vehicles"].mean()
    delta = (value - typical) / typical * 100 if typical else 0
    direction = "above" if delta >= 0 else "below"
    pct = min(value / j1["Vehicles"].max(), 1) * 100

    st.markdown(
        f'<div class="result"><div class="result-label">Predicted vehicles at '
        f'{pred_time:%H:%M} on {pred_date:%a %d %b %Y}</div>'
        f'<div><span class="big-number">{int(round(value)):,}</span>'
        f'<span class="pill" style="background:{color}22;color:{color}">{level} traffic</span></div>'
        f'<div class="compare">{abs(delta):.0f}% {direction} the usual {DAY_NAMES[pred_date.weekday()]} '
        f"at this hour ({typical:,.0f} vehicles).</div>"
        f'<div class="meter"><div style="width:{pct:.0f}%;background:{color}"></div></div></div>',
        unsafe_allow_html=True,
    )

    st.subheader("The day around it")
    fig, ax = plt.subplots(figsize=(12, 4))
    day_hist = j1[j1["DateTime"].dt.date == pred_date]
    if not day_hist.empty:
        ax.plot(day_hist["hour"], day_hist["Vehicles"], color=C["muted"], marker="o", ms=4,
                lw=1.8, label="Recorded")
    ax.plot(range(24), curve, color=C["amber"], lw=2.4, label="Model estimate")
    ax.axvline(pred_time.hour, color=C["text"], ls="--", lw=1, alpha=0.6)
    ax.scatter([pred_time.hour], [curve[pred_time.hour]], s=90, color=C["amber"], zorder=5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    show_fig(fig)

    if day_hist.empty:
        st.caption("No recorded data exists for this date, so only the model estimate is shown.")


# ============================================================
# EXPLORE
# ============================================================
def show_data_exploration_page():
    st.title("Explore data")
    st.caption("Filter a junction and date range, then look at trends, patterns and correlations.")

    raw = load_data()
    if raw is None:
        missing_file_error("Traffic dataset", DATA_PATH)
        return

    f1, f2, f3 = st.columns([1, 2, 2])
    junction = f1.selectbox("Junction", sorted(raw["Junction"].unique()))
    df = junction_frame(junction)
    min_d, max_d = df["DateTime"].min().date(), df["DateTime"].max().date()
    start = f2.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="ex_start")
    end = f3.date_input("End date", value=min(max_d, start + timedelta(days=30)),
                        min_value=start, max_value=max_d, key="ex_end")

    view = df[(df["DateTime"].dt.date >= start) & (df["DateTime"].dt.date <= end)]
    if view.empty:
        st.warning("No records in this range. Widen the dates.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(view):,}")
    m2.metric("Average", f"{view['Vehicles'].mean():,.0f}")
    m3.metric("Maximum", f"{view['Vehicles'].max():,.0f}")
    m4.metric("Minimum", f"{view['Vehicles'].min():,.0f}")
    st.write("")

    tab_trend, tab_pattern, tab_corr, tab_data = st.tabs(
        ["Trend", "Patterns", "Correlation", "Records"]
    )

    with tab_trend:
        series = view.set_index("DateTime")["Vehicles"]
        daily = (end - start).days > 31
        if daily:
            series = series.resample("D").mean()
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(series.index, series.values, color=C["amber"], lw=1.6)
        ax.fill_between(series.index, series.values, color=C["amber"], alpha=0.12)
        ax.set_ylabel("Vehicles" + (" (daily average)" if daily else ""))
        ax.grid(True, axis="y")
        fig.autofmt_xdate()
        show_fig(fig)

    with tab_pattern:
        a, b = st.columns(2)
        with a:
            st.markdown("**By hour**")
            hourly = view.groupby("hour")["Vehicles"].mean()
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(hourly.index, hourly.values, color=C["amber"], width=0.75)
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Avg vehicles")
            ax.grid(True, axis="y")
            show_fig(fig)
        with b:
            st.markdown("**By weekday**")
            weekly = view.groupby("dayofweek")["Vehicles"].mean().reindex(range(7))
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(DAY_NAMES, weekly.values, color=C["green"], width=0.65)
            ax.set_ylabel("Avg vehicles")
            ax.grid(True, axis="y")
            show_fig(fig)

        st.markdown("**Hour and weekday together**")
        grid = view.pivot_table(index="dayofweek", columns="hour", values="Vehicles", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 3.6))
        sns.heatmap(grid, cmap="YlOrBr", ax=ax, yticklabels=[DAY_NAMES[i] for i in grid.index],
                    cbar_kws={"label": "Avg vehicles"}, rasterized=True)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
        show_fig(fig)

    with tab_corr:
        corr = df[FEATURES + ["Vehicles"]].corr()
        fig, ax = plt.subplots(figsize=(9, 6.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                    linewidths=0.5, linecolor=C["bg"], annot_kws={"size": 9})
        st.caption("Correlation of each time feature with vehicle count, across the full junction history.")
        show_fig(fig)

    with tab_data:
        table = view[["DateTime", "Junction", "Vehicles"]]
        st.dataframe(table, use_container_width=True, height=420, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode(),
                           file_name=f"junction_{junction}_{start}_{end}.csv", mime="text/csv")


# ============================================================
# MAIN
# ============================================================
def main():
    st.session_state.setdefault("page", PAGES[0])
    page = sidebar()
    if page == "Dashboard":
        show_home_page()
    elif page == "Predict traffic":
        show_predict_page()
    else:
        show_data_exploration_page()


if __name__ == "__main__":
    main()
