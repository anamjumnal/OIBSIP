
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import base64
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Unemployment in India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# BACKGROUND IMAGE
# ============================================================

BACKGROUND_IMAGE = Path(__file__).parent / "unemployment_background.png"

def image_data_uri(path):
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""

_bg_uri = image_data_uri(BACKGROUND_IMAGE)

# ============================================================
# THEME
# ============================================================

BURGUNDY = "#641B35"
BURGUNDY_DARK = "#35101F"
BURGUNDY_DEEP = "#210812"
BURGUNDY_MID = "#873A58"
ROSE = "#E58AA4"
BLUSH = "#F8DCE6"
GOLD = "#F1C75B"
TEAL = "#49C7B7"
CORAL = "#E76F51"
NAVY = "#17324D"
TEAL_DARK = "#147D73"
ORANGE = "#D97706"
PURPLE_DARK = "#6B4C8A"
GREEN_DARK = "#3F6B4F"
BLUE = "#245A8D"
PURPLE = "#A88BE0"
GREEN = "#86C98A"
CREAM = "#FFF8F2"
WHITE = "#FFFFFF"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp {{
        background:
            linear-gradient(rgba(255,248,242,.22), rgba(255,248,242,.28)),
            url("{_bg_uri}") center top / cover fixed no-repeat;
        color: {BURGUNDY};
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .03;
        background-image:
            radial-gradient(circle, rgba(255,255,255,.65) 1px, transparent 1px);
        background-size: 28px 28px;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
        z-index: 0;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BURGUNDY_DARK}, #4A1027 60%, #5A1732);
        border-right: 1px solid rgba(255,255,255,.12);
    }}

    section[data-testid="stSidebar"] * {{
        color: {CREAM} !important;
    }}

    .block-container {{
        padding-top: 3.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }}

    h1, h2, h3 {{
        color: {BURGUNDY_DARK} !important;
    }}

    p, label, .stMarkdown {{
        color: {BURGUNDY_DARK};
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        min-height: 330px;
        padding: 46px 48px 42px;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 30px;
        background: rgba(255, 250, 245, .86);
        box-shadow: 0 18px 55px rgba(74, 35, 35, .18);
        margin-bottom: 24px;
    }}

    .hero::before {{
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        right: 8%;
        top: -130px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(224,180,92,.28), transparent 65%);
    }}

    .hero-content {{
        position: relative;
        z-index: 2;
        max-width: 720px;
    }}

    .eyebrow {{
        display: inline-block;
        background: rgba(224,180,92,.16);
        color: #F5D68A;
        border: 1px solid rgba(224,180,92,.42);
        padding: 7px 13px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }}

    .hero-title {{
        font-family: 'Playfair Display', serif;
        font-size: clamp(38px, 5vw, 68px);
        line-height: 1.02;
        margin: 0;
        color: {BURGUNDY_DARK};
    }}

    .hero-subtitle {{
        color: #4B2833;
        font-size: 17px;
        line-height: 1.7;
        margin-top: 18px;
        max-width: 650px;
    }}

    .page-title {{
        font-family: 'Playfair Display', serif;
        font-size: 38px;
        margin-bottom: 4px;
    }}

    .page-subtitle {{
        color: #68404B;
        margin-bottom: 22px;
    }}

    .nav-label {{
        color: #F5D68A;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 6px 0 8px;
    }}

    .card {{
        background: rgba(255,247,242,.09);
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
        backdrop-filter: blur(10px);
    }}

    .metric-card {{
        min-height: 125px;
        background: linear-gradient(145deg, rgba(125,41,72,.95), rgba(58,13,30,.92));
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 14px 35px rgba(0,0,0,.22);
    }}

    .metric-label {{
        color: #EFCED8;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .7px;
    }}

    .metric-value {{
        color: white;
        font-size: 31px;
        font-weight: 800;
        margin-top: 9px;
    }}

    .metric-note {{
        color: #DDB7C4;
        font-size: 12px;
        margin-top: 4px;
    }}

    .info-box {{
        border-left: 4px solid {GOLD};
        background: rgba(255,250,245,.90);
        border-radius: 14px;
        padding: 16px 18px;
        margin: 12px 0;
    }}

    .info-box.teal {{ border-left-color: {TEAL}; }}
    .info-box.rose {{ border-left-color: {ROSE}; }}
    .info-box.coral {{ border-left-color: {CORAL}; }}

    .pill {{
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(201,107,136,.18);
        border: 1px solid rgba(201,107,136,.35);
        color: #FFD9E4;
        font-size: 12px;
        margin: 3px;
    }}

    /* ------------------------------------------------------------------
       Streamlit / BaseWeb controls: ALWAYS use a light control surface.
       This is intentionally explicit so browser/Streamlit dark-mode rules
       cannot turn the region search/dropdown black.
       ------------------------------------------------------------------ */
    div[data-baseweb="select"],
    div[data-baseweb="input"] {{
        color-scheme: light !important;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: #FFFDF9 !important;
        border: 1px solid rgba(100,27,53,.32) !important;
        color: {BURGUNDY_DARK} !important;
        box-shadow: none !important;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] [role="combobox"],
    input {{
        background: transparent !important;
        color: {BURGUNDY_DARK} !important;
        -webkit-text-fill-color: {BURGUNDY_DARK} !important;
        caret-color: {BURGUNDY_DARK} !important;
    }}

    div[data-baseweb="select"] input::placeholder,
    div[data-baseweb="input"] input::placeholder,
    input::placeholder {{
        color: #765563 !important;
        -webkit-text-fill-color: #765563 !important;
        opacity: 1 !important;
    }}

    /* Selected chips in multiselect */
    div[data-baseweb="select"] [data-baseweb="tag"] {{
        background: #F3DCE5 !important;
        color: {BURGUNDY_DARK} !important;
        border: 1px solid #D8AFC0 !important;
    }}

    div[data-baseweb="select"] [data-baseweb="tag"] span,
    div[data-baseweb="select"] [data-baseweb="tag"] svg {{
        color: {BURGUNDY_DARK} !important;
        fill: {BURGUNDY_DARK} !important;
    }}

    div[data-baseweb="select"] svg {{
        fill: {BURGUNDY_DARK} !important;
        color: {BURGUNDY_DARK} !important;
    }}

    /* Dropdown / search popover and every option */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] > div,
    div[role="listbox"] {{
        background: #FFFDF9 !important;
        color: {BURGUNDY_DARK} !important;
        color-scheme: light !important;
        border-color: #D8B8C5 !important;
    }}

    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    div[role="listbox"] *,
    li[role="option"] {{
        color: {BURGUNDY_DARK} !important;
        -webkit-text-fill-color: {BURGUNDY_DARK} !important;
    }}

    li[role="option"],
    [data-baseweb="menu"] li {{
        background: #FFFDF9 !important;
    }}

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover {{
        background: #F3DCE5 !important;
        color: {BURGUNDY_DARK} !important;
    }}

    .stButton > button {{
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.22);
        background: linear-gradient(135deg, {BURGUNDY_MID}, {ROSE});
        color: white;
        font-weight: 800;
        padding: 10px 14px;
        transition: .2s;
    }}

    .stButton > button:hover {{
        border-color: {GOLD};
        transform: translateY(-1px);
    }}

    .stDownloadButton > button {{
        width: 100%;
        border-radius: 12px;
        background: {GOLD};
        color: {BURGUNDY_DEEP};
        border: none;
        font-weight: 800;
    }}

    [data-testid="stDataFrame"] {{
        border-radius: 16px;
        overflow: hidden;
    }}

    hr {{
        border-color: rgba(255,255,255,.13) !important;
    }}

    .small-note {{
        color: #D9B7C4;
        font-size: 12px;
    }}

    .section-heading {{
        font-size: 23px;
        font-weight: 800;
        margin: 8px 0 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Unemployment_in_India.csv")
    except FileNotFoundError:
        return None

    data.columns = data.columns.str.strip()
    data = data.rename(
        columns={
            "Estimated Unemployment Rate (%)": "Unemployment Rate",
            "Estimated Labour Participation Rate (%)":
                "Estimated Labour Participation Rate",
        }
    )

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")

    numeric_cols = [
        "Unemployment Rate",
        "Estimated Employed",
        "Estimated Labour Participation Rate",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # The Kaggle dataset does not contain an explicit Employment Rate column.
    # Derive employment-to-population rate from labour participation and unemployment:
    # Employment Rate = LPR × (1 − unemployment rate / 100).
    if (
        "Estimated Labour Participation Rate" in data.columns
        and "Unemployment Rate" in data.columns
    ):
        data["Employment Rate"] = data["Estimated Labour Participation Rate"] * (
            1 - data["Unemployment Rate"] / 100
        )

    data = data.dropna(subset=["Unemployment Rate"]).sort_values("Date")
    return data


df = load_data()

if df is None:
    st.markdown(
        """
        <div class="card">
        <h2>Dataset not found</h2>
        <p>Place <b>Unemployment_in_India.csv</b> in the same folder as this Streamlit file, then run the app again.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

region_col = "Region" if "Region" in df.columns else None

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## 🇮🇳 Unemployment India")
st.sidebar.markdown('<div class="nav-label">Explore</div>', unsafe_allow_html=True)

pages = {
    "🏠 Home": "home",
    "📊 Overview": "overview",
    "📈 Trends & EDA": "trends",
    "🦠 COVID-19 Impact": "covid",
    "🗺️ Regional Analysis": "regional",
    "📋 Data Explorer": "data",
}

page_label = st.sidebar.radio(
    "Navigation",
    list(pages.keys()),
    label_visibility="collapsed",
)
page = pages[page_label]

st.sidebar.divider()
st.sidebar.markdown('<div class="nav-label">Date Filter</div>', unsafe_allow_html=True)

date_min = df["Date"].min().date()
date_max = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

filtered = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
].copy()

st.sidebar.divider()
st.sidebar.markdown(
    '<div class="small-note">Region selection is available inside <b>Trends & EDA</b> and <b>Regional Analysis</b> so the purpose of each choice is clear.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# PLOT HELPERS
# ============================================================

# Charts sit on a light panel so labels, axes, legends and data remain readable
# over the illustrated page background.
PLOT_BG = "#FFFDF9"
PAPER_BG = "#FFFDF9"
FONT = "#35101F"
GRID = "rgba(53,16,31,.16)"

def style_fig(fig, height=430):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT, family="DM Sans"),
        title_font=dict(color=FONT, size=20),
        xaxis=dict(
            color=FONT,
            title_font=dict(color=FONT, size=14),
            tickfont=dict(color=FONT, size=12),
            gridcolor=GRID,
            zerolinecolor=GRID,
            linecolor="#8A6470",
            linewidth=1,
        ),
        yaxis=dict(
            color=FONT,
            title_font=dict(color=FONT, size=14),
            tickfont=dict(color=FONT, size=12),
            gridcolor=GRID,
            zerolinecolor=GRID,
            linecolor="#8A6470",
            linewidth=1,
        ),
        legend=dict(
            bgcolor="#FFFDF9",
            bordercolor="#D8B8C5",
            borderwidth=1,
            font=dict(color=FONT, size=12),
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color=FONT),
            title=dict(font=dict(color=FONT)),
            outlinecolor="#8A6470",
        ),
        hoverlabel=dict(
            bgcolor="#FFF8F2",
            bordercolor="#D8B8C5",
            font=dict(color="#35101F", family="DM Sans"),
        ),
        margin=dict(l=20, r=20, t=65, b=20),
    )
    return fig


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filtered_message():
    if filtered.empty:
        st.markdown(
            """
            <div class="info-box coral">
            <b>No records match the current filters.</b><br>
            Widen the date range from the sidebar or adjust the page-specific region selector.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return True
    return False


# ============================================================
# HOME
# ============================================================

if page == "home":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="eyebrow">INDIA • EMPLOYMENT • EXPLORATION</div>
                <div class="hero-title">Unemployment<br>in India</div>
                <div class="hero-subtitle">
                    Explore unemployment patterns across Indian regions, understand
                    monthly trends, compare regional differences, and examine the
                    change around the COVID-19 period through an interactive data story.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">What can you explore?</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="card"><span class="pill">01</span><h3>Overview</h3><p>See the main unemployment statistics and a quick picture of the dataset.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card"><span class="pill">02</span><h3>Trends & EDA</h3><p>Study monthly movement, selected-region trends, top regions and correlations.</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="card"><span class="pill">03</span><h3>COVID Impact</h3><p>Compare the pre-COVID and COVID-period averages and regional changes.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="card"><span class="pill">04</span><h3>Regional Analysis</h3><p>Compare average unemployment rates across regions.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card"><span class="pill">05</span><h3>Data Explorer</h3><p>Filter, sort, inspect and download the underlying records.</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="card"><span class="pill">LIVE</span><h3>{len(df):,} Records</h3><p>Use the sidebar filters to make every analysis page respond to your selection.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box teal">
        <b>How to use this website</b><br>
        Use the navigation on the left to move between pages. The date filter applies
        globally; region selectors appear only on the analysis page where they are used.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# OVERVIEW
# ============================================================

elif page == "overview":
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A quick view of unemployment levels in the currently selected data.</div>',
        unsafe_allow_html=True,
    )

    if not filtered_message():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Average rate", f"{filtered['Unemployment Rate'].mean():.2f}%", "mean unemployment rate")
        with c2:
            metric_card("Highest rate", f"{filtered['Unemployment Rate'].max():.2f}%", "highest recorded value")
        with c3:
            metric_card("Lowest rate", f"{filtered['Unemployment Rate'].min():.2f}%", "lowest recorded value")
        with c4:
            metric_card("Records", f"{len(filtered):,}", "records after filters")

        st.markdown("<br>", unsafe_allow_html=True)

        if region_col:
            regional = (
                filtered.groupby(region_col)["Unemployment Rate"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .sort_values()
            )
            fig = px.bar(
                x=regional.values,
                y=regional.index,
                orientation="h",
                color=regional.values,
                color_continuous_scale=[
                    [0, NAVY],
                    [.5, BURGUNDY],
                    [1, ORANGE],
                ],
                labels={"x": "Average Unemployment Rate (%)", "y": ""},
                title="Top 10 Regions by Average Unemployment Rate",
            )
            style_fig(fig, 480)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="info-box">
            <b>Reading this page:</b> the figures above respond to the global date
            filter. Region-specific comparisons are available on the Trends & EDA
            and Regional Analysis pages.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# TRENDS & EDA
# ============================================================

elif page == "trends":
    st.markdown('<div class="page-title">Trends & EDA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Explore how unemployment changes over time and compare specific regions without changing the rest of the website.</div>',
        unsafe_allow_html=True,
    )

    trend_regions = []
    if region_col:
        trend_regions = st.multiselect(
            "Compare regions",
            sorted(filtered[region_col].dropna().unique().tolist()),
            default=[],
            help="Choose 1–5 regions. This selector only controls the regional comparison chart below.",
        )
        if len(trend_regions) > 5:
            st.warning("Please keep the comparison to 5 regions for a readable chart.")
            trend_regions = trend_regions[:5]

    if not filtered_message():
        if region_col:
            monthly = (
                filtered.groupby(filtered["Date"].dt.to_period("M"))["Unemployment Rate"]
                .mean()
                .reset_index()
            )
            monthly["Date"] = monthly["Date"].dt.to_timestamp()

            fig = px.line(
                monthly,
                x="Date",
                y="Unemployment Rate",
                markers=True,
                title="Month-wise Average Unemployment Rate",
            )
            fig.update_traces(line=dict(color=BURGUNDY, width=4), marker=dict(color=NAVY, size=7), hovertemplate="Date: %{x|%b %Y}<br>Rate: %{y:.2f}%<extra></extra>")
            style_fig(fig, 430)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="info-box teal"><b>Observation:</b> This chart shows the monthly average unemployment rate across all regions in the current date range.</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-heading">Compare selected regions</div>', unsafe_allow_html=True)

            if trend_regions:
                comparison_regions = trend_regions
            else:
                comparison_regions = (
                    filtered.groupby(region_col)["Unemployment Rate"]
                    .mean()
                    .nlargest(3)
                    .index
                    .tolist()
                )

            if len(comparison_regions) > 8:
                comparison_regions = comparison_regions[:8]

            trend = filtered[filtered[region_col].isin(comparison_regions)].copy()

            fig2 = px.line(
                trend.sort_values("Date"),
                x="Date",
                y="Unemployment Rate",
                color=region_col,
                markers=True,
                title="Unemployment Rate by Selected Region(s)",
                color_discrete_sequence=[
                    BURGUNDY, NAVY, TEAL_DARK, ORANGE, PURPLE_DARK, BLUE, GREEN_DARK, CORAL
                ],
            )
            style_fig(fig2, 500)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                '<div class="info-box"><b>Observation:</b> Selecting regions here changes only this comparison chart. The other pages are not filtered by this choice.</div>',
                unsafe_allow_html=True,
            )

        if (
            "Estimated Labour Participation Rate" in filtered.columns
            and "Estimated Employed" in filtered.columns
        ):
            corr_cols = [
                "Unemployment Rate",
                "Employment Rate",
                "Estimated Labour Participation Rate",
            ]
            corr = filtered[corr_cols].corr()

            st.markdown('<div class="section-heading">Correlation between key metrics</div>', unsafe_allow_html=True)
            fig3 = px.imshow(
                corr,
                x=corr_cols,
                y=corr_cols,
                zmin=-1,
                zmax=1,
                color_continuous_scale=[
                    [0, "#2166AC"],
                    [.5, "#FFFDF9"],
                    [1, "#B2182B"],
                ],
                text_auto=".2f",
                title="Correlation Matrix",
            )
            style_fig(fig3, 470)
            fig3.update_traces(textfont=dict(color=FONT, size=13))
            st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# COVID
# ============================================================

elif page == "covid":
    st.markdown('<div class="page-title">COVID-19 Impact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Compare unemployment before and from March 2020 onward.</div>',
        unsafe_allow_html=True,
    )

    if not filtered_message():
        cutoff = pd.Timestamp("2020-03-01")
        pre = filtered[filtered["Date"] < cutoff]
        during = filtered[filtered["Date"] >= cutoff]

        if pre.empty or during.empty:
            st.markdown(
                """
                <div class="info-box coral">
                Your current date filter does not contain both periods. Widen the
                date range so it includes dates before and from March 2020 onward.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            pre_rate = pre["Unemployment Rate"].mean()
            covid_rate = during["Unemployment Rate"].mean()
            change = covid_rate - pre_rate
            pct = (change / pre_rate * 100) if pre_rate else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Pre-COVID", f"{pre_rate:.2f}%", "before March 2020")
            with c2:
                metric_card("COVID period", f"{covid_rate:.2f}%", "March 2020 onward")
            with c3:
                metric_card("Change", f"{change:+.2f} pp", f"{pct:+.2f}% relative change")

            comparison = pd.DataFrame(
                {
                    "Period": ["Pre-COVID", "COVID period"],
                    "Average Unemployment Rate": [pre_rate, covid_rate],
                }
            )

            fig = px.bar(
                comparison,
                x="Period",
                y="Average Unemployment Rate",
                color="Period",
                color_discrete_sequence=[TEAL_DARK, BURGUNDY],
                title="Average Unemployment Rate: Pre-COVID vs COVID Period",
                text_auto=".2f",
            )
            fig.update_traces(textfont=dict(color=FONT, size=13))
            style_fig(fig, 430)
            st.plotly_chart(fig, use_container_width=True)

            if region_col:
                pre_r = pre.groupby(region_col)["Unemployment Rate"].mean()
                covid_r = during.groupby(region_col)["Unemployment Rate"].mean()
                impact = pd.concat([pre_r, covid_r], axis=1, keys=["Pre-COVID", "COVID period"]).dropna()
                impact["Change (percentage points)"] = impact["COVID period"] - impact["Pre-COVID"]
                impact = impact.sort_values("Change (percentage points)")

                fig2 = px.bar(
                    impact.reset_index(),
                    x="Change (percentage points)",
                    y=region_col,
                    orientation="h",
                    color="Change (percentage points)",
                    color_continuous_scale=[
                        [0, "#2166AC"],
                        [.5, "#FFFDF9"],
                        [1, "#B2182B"],
                    ],
                    title="Regional Change in Average Unemployment Rate",
                )
                style_fig(fig2, 520)
                fig2.update_coloraxes(showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                """
                <div class="info-box">
                <b>Important:</b> this comparison describes differences between the
                two periods in this dataset. It does not by itself establish that
                COVID-19 was the only cause of the change.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# REGIONAL
# ============================================================

elif page == "regional":
    st.markdown('<div class="page-title">Regional Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Compare all regions, or choose specific regions below for a focused comparison.</div>',
        unsafe_allow_html=True,
    )

    regional_selection = []
    if region_col:
        regional_selection = st.multiselect(
            "Focus on regions",
            sorted(filtered[region_col].dropna().unique().tolist()),
            default=[],
            help="This selector changes only the regional charts and statistics on this page.",
        )

    regional_source = filtered.copy()
    if regional_selection:
        regional_source = regional_source[regional_source[region_col].isin(regional_selection)]

    if not filtered_message() and region_col:
        regional = (
            regional_source.groupby(region_col)
            .agg(
                Average=("Unemployment Rate", "mean"),
                Median=("Unemployment Rate", "median"),
                Minimum=("Unemployment Rate", "min"),
                Maximum=("Unemployment Rate", "max"),
                Records=("Unemployment Rate", "count"),
            )
            .round(2)
            .sort_values("Average", ascending=False)
        )

        c1, c2 = st.columns(2)

        with c1:
            top = regional.head(10).sort_values("Average")
            fig = px.bar(
                top,
                x="Average",
                y=top.index,
                orientation="h",
                color="Average",
                color_continuous_scale=[ [0, NAVY], [.5, BURGUNDY], [1, ORANGE] ],
                title="Highest Average Unemployment Rates",
                labels={"Average": "Average Rate (%)", "y": ""},
            )
            style_fig(fig, 470)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            low = regional.tail(10).sort_values("Average", ascending=True)
            fig2 = px.bar(
                low,
                x="Average",
                y=low.index,
                orientation="h",
                color="Average",
                color_continuous_scale=[ [0, TEAL_DARK], [.5, BLUE], [1, NAVY] ],
                title="Lowest Average Unemployment Rates",
                labels={"Average": "Average Rate (%)", "y": ""},
            )
            style_fig(fig2, 470)
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-heading">Regional statistics</div>', unsafe_allow_html=True)
        st.dataframe(regional, use_container_width=True)

        st.markdown(
            """
            <div class="info-box teal">
            <b>Tip:</b> use the <b>Focus on regions</b> selector above. It changes only this page, so the other pages continue showing the full filtered dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "data":
    st.markdown('<div class="page-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Inspect the filtered dataset, sort it, and download exactly what you are viewing.</div>',
        unsafe_allow_html=True,
    )

    if not filtered_message():
        q1, q2, q3 = st.columns(3)
        with q1:
            metric_card("Dataset shape", f"{df.shape[0]:,} × {df.shape[1]}", "rows × columns")
        with q2:
            metric_card("Missing values", f"{int(df.isnull().sum().sum()):,}", "in loaded dataset")
        with q3:
            metric_card("Date type", str(df["Date"].dtype), "after conversion")

        c1, c2, c3 = st.columns(3)

        with c1:
            rows = st.slider(
                "Rows to display",
                min_value=5,
                max_value=min(100, len(filtered)),
                value=min(20, len(filtered)),
            )

        with c2:
            sort_col = st.selectbox("Sort by", filtered.columns.tolist())

        with c3:
            descending = st.toggle("Descending order", value=False)

        display = filtered.sort_values(sort_col, ascending=not descending).head(rows)

        st.dataframe(display, use_container_width=True, height=520)

        csv = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download filtered CSV",
            data=csv,
            file_name="unemployment_filtered.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Rows", f"{len(filtered):,}", "after current filters")
        with c2:
            metric_card("Columns", f"{len(filtered.columns)}", "dataset fields")
        with c3:
            metric_card(
                "Average rate",
                f"{filtered['Unemployment Rate'].mean():.2f}%",
                "current selection",
            )

# ============================================================
# NO FOOTER / NO OASIS CREDIT
# ============================================================
