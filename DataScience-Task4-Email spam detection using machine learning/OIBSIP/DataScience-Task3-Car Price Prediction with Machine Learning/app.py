
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AutoPulse AI | Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car data.csv"
BG_PATH = BASE_DIR / "Gemini_Generated_Image_klofnvklofnvklof.png"


# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.rename(
        columns={
            "car_name": "name",
            "kms_driven": "km_driven",
            "fuel_type": "fuel",
        }
    )

    # Null handling
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Duplicate removal
    df = df.drop_duplicates().reset_index(drop=True)

    # Categorical cleanup
    cat_cols = ["fuel", "seller_type", "transmission", "owner"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Feature engineering
    current_year = datetime.now().year
    df["car_age"] = current_year - df["year"]

    known_brands = {
        "bajaj", "hero", "honda", "royal", "yamaha", "ktm", "tvs",
        "hyosung", "mahindra", "suzuki", "um", "maruti", "hyundai",
        "toyota", "ford", "tata", "volkswagen", "bmw", "audi",
        "renault", "nissan", "chevrolet", "mercedes-benz"
    }

    model_to_brand = {
        **dict.fromkeys(
            [
                "800", "alto 800", "alto k10", "baleno", "ciaz", "dzire",
                "ertiga", "ignis", "omni", "ritz", "s cross", "swift",
                "sx4", "vitara brezza", "wagon r"
            ],
            "Maruti",
        ),
        **dict.fromkeys(["amaze", "brio", "city", "jazz"], "Honda"),
        **dict.fromkeys(
            [
                "camry", "corolla", "corolla altis", "etios cross",
                "etios g", "etios gd", "etios liva", "fortuner",
                "innova", "land cruiser"
            ],
            "Toyota",
        ),
        **dict.fromkeys(
            ["creta", "elantra", "eon", "grand i10", "i10", "i20", "verna", "xcent"],
            "Hyundai",
        ),
        **dict.fromkeys(["activa 3g", "activa 4g"], "Honda"),
    }

    def get_brand(name):
        n = " ".join(str(name).lower().split())
        first = n.split()[0]

        if first in known_brands:
            if first == "royal":
                return "Royal Enfield"
            if first in {"ktm", "tvs", "um"}:
                return first.upper()
            return first.title()

        return model_to_brand.get(n, first.title())

    df["brand"] = df["name"].apply(get_brand)

    # Same rare-brand grouping used in the notebook
    rare = df["brand"].value_counts()[lambda s: s < 5].index
    df["brand"] = df["brand"].replace(dict.fromkeys(rare, "Other"))

    return df


@st.cache_resource
def train_models(df):
    model_df = df.drop(columns=["name", "year"])

    cat_features = ["fuel", "seller_type", "transmission", "owner", "brand"]
    model_df = pd.get_dummies(
        model_df,
        columns=cat_features,
        drop_first=True,
        dtype=int,
    )

    X = model_df.drop(columns="selling_price")
    y = model_df["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R²": r2_score(y_test, pred),
            }
        )

    results_df = pd.DataFrame(results).set_index("Model")

    # Same selection logic as the notebook: highest R²
    best_name = results_df["R²"].idxmax()
    best_model = models[best_name]

    # Refit selected model on the full dataset for website predictions
    best_model.fit(X, y)

    return (
        models,
        results_df,
        predictions,
        X,
        y,
        X_test,
        y_test,
        best_name,
        best_model,
    )


@st.cache_data
def get_bg_base64():
    if not BG_PATH.exists():
        return ""

    return base64.b64encode(BG_PATH.read_bytes()).decode()


def predict_from_inputs(
    model,
    feature_columns,
    present_price,
    km_driven,
    car_age,
    fuel,
    seller_type,
    transmission,
    owner,
    brand,
):
    # Build the row directly in the training feature space.
    # (pd.get_dummies(..., drop_first=True) on a single row drops the only category,
    #  which silently zeroes every categorical input - so the dummies are set by hand.)
    row = pd.DataFrame(0.0, index=[0], columns=list(feature_columns))

    row.loc[0, "present_price"] = present_price
    row.loc[0, "km_driven"] = km_driven
    row.loc[0, "car_age"] = car_age

    # The first (alphabetical) category of each field was dropped in training, so it is
    # represented by all zeros; every other category has its own 0/1 column.
    for prefix, value in [
        ("fuel", fuel),
        ("seller_type", seller_type),
        ("transmission", transmission),
        ("owner", str(owner)),
        ("brand", brand),
    ]:
        col = f"{prefix}_{value}"
        if col in row.columns:
            row.loc[0, col] = 1.0

    return float(model.predict(row)[0])


# ============================================================
# LOAD + TRAIN
# ============================================================
df = load_data()
(
    models,
    results_df,
    predictions,
    X,
    y,
    X_test,
    y_test,
    best_name,
    best_model,
) = train_models(df)


# ============================================================
# CUSTOM CSS
# ============================================================
bg64 = get_bg_base64()

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap');

    .stApp {{
        min-height: 100vh;
        background: #050b16;
        color: #f8fbff;
        font-family: 'Inter', sans-serif;
        position: relative;
    }}

    /* Keep the supplied 2048x1143 artwork visibly horizontal (16:9-ish),
       never rotate it, stretch it, or crop it into a portrait image. */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: -2;
        background-image:
            linear-gradient(115deg,
                rgba(5, 13, 28, 0.72) 0%,
                rgba(7, 23, 44, 0.48) 38%,
                rgba(14, 17, 31, 0.30) 67%,
                rgba(5, 10, 22, 0.58) 100%),
            url("data:image/png;base64,{bg64}");
        background-position: center top;
        background-repeat: no-repeat;
        background-size: auto 100vh;
    }}

    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: -3;
        background: #050b16;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(5, 15, 30, 0.96), rgba(11, 18, 38, 0.91));
        border-right: 1px solid rgba(255,255,255,0.10);
    }}

    [data-testid="stSidebar"] * {{
        color: #edf6ff !important;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }}

    .hero {{
        padding: 34px 38px;
        border-radius: 28px;
        background:
            linear-gradient(120deg,
                rgba(7, 17, 34, 0.90),
                rgba(8, 42, 61, 0.72),
                rgba(71, 17, 52, 0.56));
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 25px 70px rgba(0,0,0,0.35);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
        margin-bottom: 22px;
    }}

    .hero:after {{
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -90px;
        top: -110px;
        border-radius: 50%;
        background: rgba(0, 229, 255, 0.22);
        filter: blur(10px);
    }}

    .eyebrow {{
        color: #67e8f9;
        font-size: 13px;
        letter-spacing: 3px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 9px;
    }}

    .hero h1 {{
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(34px, 5vw, 66px);
        line-height: 1.03;
        margin: 0;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff, #67e8f9, #ff6b9d, #ffd166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .hero p {{
        color: #d8e9f7;
        max-width: 760px;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 15px;
    }}

    .tag {{
        display: inline-block;
        padding: 8px 13px;
        margin: 7px 6px 0 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.15);
        color: #eaf8ff;
        font-size: 12px;
        font-weight: 700;
    }}

    .metric-card {{
        padding: 21px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045));
        border: 1px solid rgba(255,255,255,0.13);
        min-height: 125px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }}

    .metric-label {{
        color: #9fb8ca;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }}

    .metric-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 27px;
        margin-top: 9px;
        color: #ffffff;
    }}

    .metric-sub {{
        color: #67e8f9;
        font-size: 12px;
        margin-top: 5px;
    }}

    .section-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 25px;
        margin: 30px 0 12px 0;
        color: #ffffff;
    }}

    .glass {{
        padding: 24px;
        border-radius: 22px;
        background: rgba(8, 19, 35, 0.72);
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(12px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.22);
    }}

    .prediction-box {{
        margin-top: 18px;
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.18), rgba(255, 77, 141, 0.16), rgba(255, 209, 102, 0.14));
        border: 1px solid rgba(103, 232, 249, 0.38);
        box-shadow: 0 18px 45px rgba(0,0,0,0.28);
    }}

    .prediction-label {{
        color: #bdebf5;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 800;
    }}

    .prediction-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 46px;
        margin: 8px 0;
        background: linear-gradient(90deg, #67e8f9, #ffffff, #ffd166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .small-note {{
        color: #9fb8ca;
        font-size: 12px;
        line-height: 1.6;
    }}

    div.stButton > button {{
        width: 100%;
        border: 0;
        border-radius: 14px;
        padding: 13px 18px;
        font-weight: 800;
        color: #06111d;
        background: linear-gradient(90deg, #67e8f9, #4ade80, #ffd166);
        box-shadow: 0 10px 25px rgba(0,229,255,0.18);
        transition: transform .18s ease, box-shadow .18s ease;
    }}

    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(0,229,255,0.30);
    }}

    .footer {{
        margin-top: 45px;
        padding: 18px;
        text-align: center;
        color: #8fa7b9;
        font-size: 12px;
    }}

    .stSelectbox label, .stNumberInput label, .stTextInput label {{
        color: #dceeff !important;
        font-weight: 700 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="font-family:Orbitron;font-size:21px;font-weight:800;
                    background:linear-gradient(90deg,#67e8f9,#ff6b9d,#ffd166);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            AUTOPULSE AI
        </div>
        <div style="color:#91a9ba;font-size:12px;margin-top:4px;">
            Used Car Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏁 Overview", "🚘 Price Predictor", "📊 Model Insights", "🗃️ Dataset Explorer"],
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="small-note">
        <b>ML pipeline</b><br>
        Cleaning → Feature Engineering → Encoding → Regression → Evaluation
        <br><br>
        <b>Models</b><br>
        Linear Regression<br>
        Random Forest<br>
        Gradient Boosting
        <br><br>
        <b>Selected model by R²</b><br>
        {best_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning · Regression · Used Cars</div>
        <h1>AutoPulse AI</h1>
        <p>
            A cinematic car-price intelligence dashboard built around the
            Vehicle dataset from CarDekho. Explore the data, compare regression
            models, and generate a selling-price estimate from real vehicle features.
        </p>
        <span class="tag">Python</span>
        <span class="tag">Pandas</span>
        <span class="tag">Scikit-learn</span>
        <span class="tag">Regression</span>
        <span class="tag">Streamlit</span>
        <span class="tag">EDA</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "🏁 Overview":
    st.markdown('<div class="section-title">Dashboard snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Vehicles analysed</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-sub">cleaned dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value">{X.shape[1]}</div>
                <div class="metric-sub">after encoding</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">R² · selected model</div>
                <div class="metric-value">{results_df.loc[best_name, "R²"]:.3f}</div>
                <div class="metric-sub">{best_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{results_df.loc[best_name, "RMSE"]:.3f}</div>
                <div class="metric-sub">price units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">What the pipeline does</div>', unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="glass">
                <h3 style="color:#67e8f9;">01 · Prepare</h3>
                <p style="color:#d5e5f0;line-height:1.7;">
                    Handles missing values, removes duplicate records, standardises
                    categorical values and derives <b>car age</b> and <b>brand</b>.
                </p>
                <h3 style="color:#ff7aa8;">02 · Understand</h3>
                <p style="color:#d5e5f0;line-height:1.7;">
                    Explores price distributions, fuel-type differences, age-price
                    relationships and feature correlations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="glass">
                <h3 style="color:#ffd166;">03 · Learn</h3>
                <p style="color:#d5e5f0;line-height:1.7;">
                    Trains Linear Regression, Random Forest and Gradient Boosting
                    models using an 80/20 train-test split.
                </p>
                <h3 style="color:#4ade80;">04 · Predict</h3>
                <p style="color:#d5e5f0;line-height:1.7;">
                    Uses the selected regression model to estimate the selling price
                    of a used vehicle from the same engineered feature space.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)

    display_results = results_df.copy().round(3)
    st.dataframe(display_results, width="stretch")

    st.markdown(
        '<div class="small-note">Metrics are calculated on the notebook-style 20% test split with random_state=42.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTOR
# ============================================================
elif page == "🚘 Price Predictor":
    st.markdown('<div class="section-title">Estimate a used-car selling price</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass">
            <div style="color:#dceeff;line-height:1.7;">
                Enter the vehicle details below. The app applies the same
                feature engineering and one-hot encoding logic used by the
                machine-learning notebook.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    brands = sorted(df["brand"].unique().tolist())
    fuels = sorted(df["fuel"].unique().tolist())
    sellers = sorted(df["seller_type"].unique().tolist())
    transmissions = sorted(df["transmission"].unique().tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", brands, index=None, placeholder="Choose a brand")
        year = st.selectbox("Manufacturing year", list(range(datetime.now().year, int(df["year"].min()) - 1, -1)), index=None, placeholder="Choose a year")
        present_price = st.number_input("Present price (₹ lakh)", min_value=0.1, max_value=100.0, value=None, placeholder="Enter present price", step=0.1)

    with col2:
        km_driven = st.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=None, placeholder="Enter kilometres driven", step=1_000)
        fuel = st.selectbox("Fuel type", fuels, index=None, placeholder="Choose fuel type")
        seller_type = st.selectbox("Seller type", sellers, index=None, placeholder="Choose seller type")

    with col3:
        transmission = st.selectbox("Transmission", transmissions, index=None, placeholder="Choose transmission")
        owner = st.selectbox("Previous owners", sorted(df["owner"].astype(str).unique().tolist()), index=None, placeholder="Choose previous owners")
        car_age = max(datetime.now().year - int(year), 0) if year is not None else None

        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:28px;">
                <div class="metric-label">Calculated car age</div>
                <div class="metric-value">{car_age} yrs</div>
                <div class="metric-sub">automatically derived from year</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if st.button("⚡ CALCULATE ESTIMATED SELLING PRICE"):
        if any(v is None for v in [brand, year, present_price, km_driven, fuel, seller_type, transmission, owner]):
            st.warning("Please choose/enter all vehicle details before calculating the price.")
            st.stop()

        predicted = predict_from_inputs(
            best_model,
            X.columns,
            present_price,
            km_driven,
            car_age,
            fuel,
            seller_type,
            transmission,
            owner,
            brand,
        )

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimated selling price</div>
                <div class="prediction-value">₹ {predicted:,.2f} Lakh</div>
                <div style="color:#ccecf4;font-size:13px;">
                    Generated using <b>{best_name}</b> trained on the uploaded dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This is a machine-learning estimate based on the dataset and selected features; "
            "it is not a live market quote."
        )


# ============================================================
# MODEL INSIGHTS
# ============================================================
elif page == "📊 Model Insights":
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, model_name in zip((m1, m2, m3), models.keys()):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{model_name}</div>
                    <div class="metric-value">R² {results_df.loc[model_name, "R²"]:.3f}</div>
                    <div class="metric-sub">
                        MAE {results_df.loc[model_name, "MAE"]:.3f}
                        · RMSE {results_df.loc[model_name, "RMSE"]:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Actual vs predicted</div>', unsafe_allow_html=True)

    chart_df = pd.DataFrame({"Actual": y_test.reset_index(drop=True)})

    for model_name in models.keys():
        chart_df[model_name] = predictions[model_name]

    st.scatter_chart(
        chart_df,
        x="Actual",
        y=best_name,
        width="stretch",
    )

    st.markdown('<div class="section-title">Feature importance / influence</div>', unsafe_allow_html=True)

    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=X.columns)
    else:
        importance = pd.Series(np.abs(best_model.coef_), index=X.columns)

    top = importance.sort_values(ascending=False).head(12).sort_values()

    importance_df = pd.DataFrame(
        {"Feature": top.index, "Importance": top.values}
    ).set_index("Feature")

    st.bar_chart(importance_df, width="stretch")

    st.markdown(
        f"""
        <div class="glass">
            <b style="color:#67e8f9;">Selected model:</b> {best_name}<br><br>
            The selection follows the notebook's R²-based model-selection logic.
            The feature chart uses absolute linear coefficients when the selected
            model is Linear Regression, and native feature importance for tree models.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET
# ============================================================
else:
    st.markdown('<div class="section-title">Dataset explorer</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows", len(df))
    with c2:
        st.metric("Columns", len(df.columns))
    with c3:
        st.metric("Median selling price", f"₹ {df['selling_price'].median():.2f} L")

    st.dataframe(
        df[
            [
                "name",
                "brand",
                "year",
                "car_age",
                "selling_price",
                "present_price",
                "km_driven",
                "fuel",
                "seller_type",
                "transmission",
                "owner",
            ]
        ],
        width="stretch",
        height=520,
    )

    st.markdown('<div class="section-title">Quick dataset facts</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)

    with f1:
        st.write("**Fuel distribution**")
        st.bar_chart(df["fuel"].value_counts())

    with f2:
        st.write("**Top brands by vehicle count**")
        st.bar_chart(df["brand"].value_counts().head(10))

    st.markdown(
        '<div class="small-note">The dataset shown here is the same CSV used for model training.</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="footer">
        AutoPulse AI · Car Price Prediction with Machine Learning ·
        Python + Pandas + Scikit-learn + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
