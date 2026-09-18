# =============================================================================
# Indian Road Accident Severity Dashboard
# Streamlit multi-page application — 4 pages only
# Run: streamlit run app.py
# =============================================================================

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib
import warnings
import os
import time

# sklearn imports — used when building a fresh model as fallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = os.path.join(_HERE, "data", "processed", "crash_level_accidents.csv")
MODEL_PATH = os.path.join(_HERE, "models", "best_model_bundle.joblib")

TARGET_COL = "crash_severity_first"

SEVERITY_ORDER  = ["minor", "major", "fatal"]
SEVERITY_COLORS = {"minor": "#2ecc71", "major": "#f39c12", "fatal": "#e74c3c"}

# Folium marker colours (must be valid Folium colour names)
FOLIUM_COLORS = {"minor": "green", "major": "orange", "fatal": "red"}

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Road Accident Severity",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tighten top padding */
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* Severity badges */
    .severity-badge {
        display: inline-block; padding: 10px 28px; border-radius: 30px;
        font-size: 1.4rem; font-weight: 900; color: #fff; letter-spacing: 2px;
        text-transform: uppercase; margin: 4px 0;
    }
    .badge-minor  { background: linear-gradient(135deg,#2ecc71,#27ae60); }
    .badge-major  { background: linear-gradient(135deg,#f39c12,#e67e22); }
    .badge-fatal  { background: linear-gradient(135deg,#e74c3c,#c0392b); }

    /* Info highlight boxes */
    .info-card {
        background: #f8faff;
        border-left: 5px solid #4a6fa5;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        line-height: 1.6;
    }
    .stat-card {
        background: linear-gradient(135deg,#667eea,#764ba2);
        border-radius: 10px;
        padding: 18px;
        color: white;
        text-align: center;
        margin: 4px;
    }
    .stat-card h2 { font-size: 2rem; margin: 0; }
    .stat-card p  { font-size: 0.85rem; margin: 4px 0 0; opacity: 0.9; }

    /* Section dividers */
    hr { margin: 18px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── SESSION-STATE DEFAULTS ────────────────────────────────────────────────────
def _init_state():
    """Initialise session state keys once."""
    defaults = {
        "df":            None,      # Main DataFrame
        "model_bundle":  None,      # Loaded joblib bundle dict
        "dataset_seed":  42,        # Seed for random sample on Dataset page
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── CACHED LOADERS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and cache the crash-level CSV from disk."""
    return pd.read_csv(DATA_PATH)


# Known categorical columns used during original model training
_CAT_COLS = [
    "city_name_first", "state_name_first", "weekday_name_first",
    "route_category_first", "weather_condition_first", "visibility_level_first",
    "congestion_level_first", "road_surface_condition_first",
    "primary_cause_first", "dominant_vehicle_type_first",
]
_DROP_COLS = [
    "crash_ref_id", "incident_time_first", "festival_name_first",
    "lat_coord_first", "lon_coord_first",
]


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    """
    Try to load the pre-trained model bundle from disk.
    If that fails (e.g., sklearn version mismatch), train a fresh
    Random Forest from the CSV and return an equivalent bundle.
    """
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # ── Fallback: train fresh from the dataset ──────────────────────────
        df_raw = pd.read_csv(DATA_PATH)
        df     = df_raw.copy()
        df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True, errors="ignore")

        X = df.drop(columns=[TARGET_COL], errors="ignore")
        y = df[TARGET_COL]

        le     = LabelEncoder()
        y_enc  = le.fit_transform(y)

        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in _CAT_COLS if c in X.columns]
        used_cols = num_cols + cat_cols
        X = X[used_cols]

        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=15, random_state=42,
                n_jobs=-1, class_weight="balanced",
            )),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc    = round(accuracy_score(y_test, y_pred) * 100, 2)

        # Feature importance
        importances = pipeline.named_steps["clf"].feature_importances_
        feat_imp = pd.DataFrame({"Feature": used_cols, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
        feat_imp["Rank"] = feat_imp.index + 1

        # Metadata for prediction widgets
        feat_ranges  = {}
        feat_options = {}
        for col in num_cols:
            col_data = X[col].dropna()
            feat_ranges[col] = {
                "min":    float(col_data.min()),
                "max":    float(col_data.max()),
                "median": float(col_data.median()),
            }
        for col in cat_cols:
            feat_options[col] = sorted(df[col].dropna().unique().tolist())

        return {
            "pipeline":      pipeline,
            "label_encoder": le,
            "feature_names": used_cols,
            "num_cols":      num_cols,
            "cat_cols":      cat_cols,
            "feat_imp":      feat_imp,
            "feat_ranges":   feat_ranges,
            "feat_options":  feat_options,
            "metrics":       {"accuracy": acc},
            "_fallback":     True,   # Flag so UI can note it was freshly trained
        }


# ── PREDICTION HELPER ─────────────────────────────────────────────────────────
def predict_severity(bundle: dict, input_dict: dict):
    """
    Run inference with the loaded pipeline bundle.
    Returns (predicted_label, probas_array, class_names).
    """
    pipeline      = bundle["pipeline"]
    le            = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    # Build a one-row DataFrame aligned to the pipeline's expected columns
    row      = {col: input_dict.get(col, np.nan) for col in feature_names}
    input_df = pd.DataFrame([row])

    probas     = pipeline.predict_proba(input_df)[0]
    pred_idx   = int(np.argmax(probas))
    pred_label = le.classes_[pred_idx]
    return pred_label, probas, le.classes_


# ── SEVERITY BADGE HTML ───────────────────────────────────────────────────────
def severity_badge(label: str) -> str:
    return (
        f'<span class="severity-badge badge-{label.lower()}">'
        f'{label.upper()}</span>'
    )


# ── AUTO-LOAD DATA & MODEL ────────────────────────────────────────────────────
if st.session_state.df is None:
    try:
        st.session_state.df = load_data()
    except Exception:
        pass  # Will show error on relevant pages

if st.session_state.model_bundle is None:
    try:
        st.session_state.model_bundle = load_model_bundle()
    except Exception:
        pass  # Will show error on Predict page


# =============================================================================
# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
# =============================================================================
with st.sidebar:
    st.markdown("## 🚦 Road Accident")
    st.markdown("### Severity Dashboard")
    st.markdown("*India — ML Prediction App*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home", "📂  Dataset", "🔮  Predict Severity", "🗺️  India Accident Map"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick status indicators
    df_ok    = st.session_state.df is not None
    model_ok = st.session_state.model_bundle is not None

    st.markdown(f"**Dataset:** {'✅ Loaded' if df_ok else '❌ Not found'}")

    if model_ok:
        is_fallback = st.session_state.model_bundle.get("_fallback", False)
        model_label = "✅ Ready (fresh)" if is_fallback else "✅ Ready"
        acc = st.session_state.model_bundle.get("metrics", {}).get("accuracy", "—")
        st.markdown(f"**Model:**   {model_label}")
        st.markdown(f"**Accuracy:** `{acc}%`")
    else:
        st.markdown("**Model:**   ❌ Not found")

    if df_ok:
        df_global = st.session_state.df
        st.markdown(f"**Records:** `{len(df_global):,}`")

    st.markdown("---")
    st.caption("Indian Road Accident Severity\nv3.0 · Streamlit + scikit-learn")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 1 — HOME
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
if page == "🏠  Home":

    # ── HERO BANNER ──────────────────────────────────────────────────────────
    st.title("🚦 Indian Road Accident Severity Dashboard")
    st.markdown(
        "##### Predict, analyse, and visualise road accident severity across India — "
        "powered by a **Random Forest** classifier trained on real crash data."
    )
    st.markdown("---")

    # ── PROBLEM STATEMENT (expandable) ───────────────────────────────────────
    with st.expander("📋  Problem Statement — Click to read", expanded=True):
        col_ps, col_img = st.columns([3, 1])
        with col_ps:
            st.markdown(
                """
**India records nearly 500,000 road accidents every year**, making it one of the
highest accident-prone countries in the world. In 2023 alone, over **1.5 lakh lives
were lost** on Indian roads — more than **400 fatalities per day**.

This application addresses a critical public-safety challenge: **given the conditions
surrounding an accident (weather, road type, time of day, vehicle type, etc.), can we
predict how severe the outcome will be?**

By identifying high-risk patterns early, authorities can:
- Prioritise emergency response resources
- Design smarter road-safety interventions
- Educate the public about risk factors
                """
            )
        with col_img:
            st.metric("Annual Accidents", "4.8 Lakh+", "+2% YoY")
            st.metric("Annual Fatalities", "1.68 Lakh+", "40/hr")
            st.metric("Economic Cost", "₹1.47 Lakh Cr.", "3.14% GDP")

    st.markdown("---")

    # ── KEY STATS FROM DATASET ────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("📊 Dataset at a Glance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📋 Total Records",  f"{len(df):,}")
        c2.metric("📌 Features",        f"{df.shape[1]}")
        if "city_name_first"  in df.columns:
            c3.metric("🏙️ Cities",  f"{df['city_name_first'].nunique():,}")
        if "state_name_first" in df.columns:
            c4.metric("🗺️ States", f"{df['state_name_first'].nunique():,}")
        if TARGET_COL in df.columns:
            fatal_pct = (df[TARGET_COL] == "fatal").mean() * 100
            c5.metric("💀 Fatal Rate", f"{fatal_pct:.1f}%")

        st.markdown("---")

        # Severity distribution + donut in two columns
        col_bar, col_pie = st.columns([3, 2])

        with col_bar:
            st.subheader("🎯 Severity Distribution")
            if TARGET_COL in df.columns:
                counts = (
                    df[TARGET_COL]
                    .value_counts()
                    .reindex(SEVERITY_ORDER, fill_value=0)
                    .reset_index()
                )
                counts.columns = ["Severity", "Count"]
                counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
                fig = px.bar(
                    counts, x="Severity", y="Count", color="Severity",
                    color_discrete_map=SEVERITY_COLORS,
                    text=counts["Pct"].apply(lambda x: f"{x}%"),
                    height=320,
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10),
                                  yaxis_title="Number of Crashes")
                st.plotly_chart(fig, use_container_width=True)

        with col_pie:
            st.subheader("🥧 Class Share")
            if TARGET_COL in df.columns:
                pie_data = df[TARGET_COL].value_counts().reset_index()
                pie_data.columns = ["Severity", "Count"]
                fig2 = px.pie(
                    pie_data, values="Count", names="Severity",
                    color="Severity", color_discrete_map=SEVERITY_COLORS,
                    hole=0.48, height=320,
                )
                fig2.update_layout(margin=dict(t=10, b=10),
                                   legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── PROJECT OBJECTIVES ────────────────────────────────────────────────────
    st.subheader("🎯 Project Objectives")
    obj_col1, obj_col2 = st.columns(2)
    with obj_col1:
        st.markdown(
            """
<div class="info-card">
<b>1 · Predictive Modelling</b><br>
Build a multi-class classifier to predict accident severity
(<em>Minor / Major / Fatal</em>) from contextual features.
</div>
<div class="info-card">
<b>2 · Feature Understanding</b><br>
Identify which road, weather, vehicle, and temporal factors
most strongly predict severe outcomes.
</div>
            """,
            unsafe_allow_html=True,
        )
    with obj_col2:
        st.markdown(
            """
<div class="info-card">
<b>3 · Geospatial Visualisation</b><br>
Map accident hotspots across India to help target
infrastructure and emergency-response improvements.
</div>
<div class="info-card">
<b>4 · Accessible Interface</b><br>
Provide a simple web interface so planners, researchers,
and policymakers can explore the data without coding.
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── METHODOLOGY OVERVIEW ──────────────────────────────────────────────────
    st.subheader("🔬 Methodology Overview")
    tab_data, tab_model, tab_eval = st.tabs(
        ["📂  Data Pipeline", "🤖  Model Architecture", "📈  Evaluation Strategy"]
    )

    with tab_data:
        st.markdown(
            """
| Step | Details |
|---|---|
| **Source** | Crash-level aggregated Indian road accident records |
| **Records** | ~20,000 crash events across 49 features |
| **Preprocessing** | Median imputation (numeric) · Mode imputation (categorical) |
| **Encoding** | Ordinal Encoder for categorical columns |
| **Target** | `crash_severity_first` → minor · major · fatal |
            """
        )

    with tab_model:
        st.markdown(
            """
| Property | Value |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Estimators** | 150 decision trees |
| **Max Depth** | 15 levels |
| **Class Weights** | Balanced (addresses class imbalance) |
| **Pipeline** | Sklearn `Pipeline` + `ColumnTransformer` |
| **Persistence** | Saved as `.joblib` for fast reload |
            """
        )

    with tab_eval:
        st.markdown(
            """
| Metric | Strategy |
|---|---|
| **Train / Test Split** | 80% / 20% stratified |
| **Primary Metric** | Weighted F1-Score |
| **Secondary Metrics** | Accuracy, Precision, Recall per class |
| **Validation** | Confusion matrix + classification report |
            """
        )

    st.markdown("---")

    # ── USAGE GUIDE ───────────────────────────────────────────────────────────
    st.subheader("🧭 How to Use This App")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(
            """
<div class="info-card" style="text-align:center">
<div style="font-size:2rem">📂</div>
<b>Dataset</b><br>
Browse 20+ random records, see column info, and refresh the sample.
</div>
            """, unsafe_allow_html=True)
    with g2:
        st.markdown(
            """
<div class="info-card" style="text-align:center">
<div style="font-size:2rem">🔮</div>
<b>Predict Severity</b><br>
Enter accident details interactively and get an instant ML prediction.
</div>
            """, unsafe_allow_html=True)
    with g3:
        st.markdown(
            """
<div class="info-card" style="text-align:center">
<div style="font-size:2rem">🗺️</div>
<b>Accident Map</b><br>
Explore geo-coded accident hotspots on an interactive map of India.
</div>
            """, unsafe_allow_html=True)
    with g4:
        st.markdown(
            """
<div class="info-card" style="text-align:center">
<div style="font-size:2rem">🎨</div>
<b>Colour Coding</b><br>
🟢 Minor &nbsp;|&nbsp; 🟠 Major &nbsp;|&nbsp; 🔴 Fatal
across all charts and maps.
</div>
            """, unsafe_allow_html=True)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 2 — DATASET
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
elif page == "📂  Dataset":

    st.title("📂 Dataset Explorer")
    st.markdown(
        "Inspect the crash-level accident dataset. A **random sample of 25 records** "
        "is shown by default; click **🔄 Refresh Sample** for a new selection."
    )
    st.markdown("---")

    # ── LOAD DATA ────────────────────────────────────────────────────────────
    if st.session_state.df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`. Please check the file path.")
        st.stop()

    df = st.session_state.df

    # ── TOP KPI STRIP ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Total Records",  f"{len(df):,}")
    k2.metric("📌 Total Features", f"{df.shape[1]}")
    if "state_name_first" in df.columns:
        k3.metric("🗺️ States", f"{df['state_name_first'].nunique()}")
    if TARGET_COL in df.columns:
        k4.metric("🏷️ Severity Classes", df[TARGET_COL].nunique())

    st.markdown("---")

    # ── RANDOM SAMPLE TABLE ───────────────────────────────────────────────────
    st.subheader("🎲 Random Sample (25 Records)")

    col_refresh, col_n, _ = st.columns([1, 2, 4])
    with col_refresh:
        if st.button("🔄 Refresh Sample", type="primary"):
            st.session_state.dataset_seed = np.random.randint(0, 99999)

    with col_n:
        n_sample = st.slider("Sample size", min_value=20, max_value=100, value=25, step=5)

    sample_df = df.sample(n=min(n_sample, len(df)), random_state=st.session_state.dataset_seed)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    st.caption(
        f"Showing {len(sample_df):,} randomly selected records from a dataset of "
        f"{len(df):,} total crashes."
    )

    st.markdown("---")

    # ── DETAILED TABS ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Statistics", "🔤  Column Info", "❓  Missing Values", "📈  Distribution"]
    )

    with tab1:
        st.subheader("Numeric Feature Summary")
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            st.dataframe(num_df.describe().T.round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab2:
        st.subheader("Column Details")
        dtype_df = pd.DataFrame({
            "Column":       df.columns,
            "Type":         df.dtypes.values.astype(str),
            "Non-Null":     df.notnull().sum().values,
            "Nulls":        df.isnull().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Example":      [
                str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else ""
                for c in df.columns
            ],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=480)

        st.markdown("---")
        st.subheader("📖 Key Column Descriptions")
        descriptions = {
            "crash_severity_first":      "Target — minor / major / fatal outcome",
            "lat_coord_first":           "Latitude of the crash location",
            "lon_coord_first":           "Longitude of the crash location",
            "state_name_first":          "Indian state where the crash occurred",
            "city_name_first":           "Nearest city to the crash",
            "hour_of_day_first":         "Hour of the day (0 – 23)",
            "weather_condition_first":   "Weather at the time of crash",
            "road_surface_condition_first": "Surface condition (dry, wet, etc.)",
            "route_category_first":      "Road type (highway, urban, rural)",
            "vehicle_count_first":       "Number of vehicles involved",
            "mean_speed_at_impact_kmph": "Average vehicle speed at impact",
            "helmet_seatbelt_usage_rate":"Fraction of occupants using safety gear",
            "alcohol_suspected_flag_first": "1 if alcohol involvement suspected",
            "primary_cause_first":       "Primary cause of the crash",
        }
        desc_df = pd.DataFrame(
            [{"Column": k, "Description": v} for k, v in descriptions.items()]
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    with tab3:
        missing_cnt = df.isnull().sum()
        miss_df = (
            pd.DataFrame({
                "Column": missing_cnt.index,
                "Count":  missing_cnt.values,
                "Pct %":  (missing_cnt.values / len(df) * 100).round(2),
            })
            .query("Count > 0")
            .reset_index(drop=True)
        )
        if miss_df.empty:
            st.success("✅ No missing values in the dataset!")
        else:
            st.warning(f"{len(miss_df)} column(s) have missing values.")
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            fig = px.bar(
                miss_df, x="Column", y="Pct %", text="Pct %",
                color="Pct %", color_continuous_scale="Reds",
                height=320,
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Severity Class Distribution")
        if TARGET_COL in df.columns:
            dist_data = (
                df[TARGET_COL]
                .value_counts()
                .reindex(SEVERITY_ORDER, fill_value=0)
                .reset_index()
            )
            dist_data.columns = ["Severity", "Count"]
            dist_data["Percentage"] = (
                dist_data["Count"] / dist_data["Count"].sum() * 100
            ).round(1)

            dc1, dc2 = st.columns(2)
            with dc1:
                fig_d = px.bar(
                    dist_data, x="Severity", y="Count", color="Severity",
                    color_discrete_map=SEVERITY_COLORS,
                    text="Percentage", height=360,
                )
                fig_d.update_traces(
                    texttemplate="%{text}%", textposition="outside"
                )
                fig_d.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_d, use_container_width=True)
            with dc2:
                st.dataframe(dist_data, use_container_width=True, hide_index=True)
                st.markdown("")
                # Accidents per state
                if "state_name_first" in df.columns:
                    st.markdown("**Top States by Accident Count**")
                    top_states = (
                        df["state_name_first"].value_counts().head(8).reset_index()
                    )
                    top_states.columns = ["State", "Count"]
                    st.dataframe(top_states, use_container_width=True, hide_index=True)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 3 — PREDICT SEVERITY
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
elif page == "🔮  Predict Severity":

    st.title("🔮 Predict Accident Severity")
    st.markdown(
        "Fill in the accident details below and click **Predict Severity** "
        "to get an instant ML-based prediction."
    )
    st.markdown("---")

    # ── MODEL CHECK ───────────────────────────────────────────────────────────
    if st.session_state.model_bundle is None:
        st.error(
            f"Pre-trained model not found at `{MODEL_PATH}`. "
            "Please ensure `models/best_model_bundle.joblib` exists."
        )
        st.stop()

    bundle = st.session_state.model_bundle

    # Extract metadata from bundle for widget construction
    feat_ranges  = bundle.get("feat_ranges",  {})
    feat_options = bundle.get("feat_options", {})
    num_cols     = bundle.get("num_cols",     [])
    cat_cols     = bundle.get("cat_cols",     [])
    all_feats    = bundle.get("feature_names", [])
    feat_imp     = bundle.get("feat_imp",     pd.DataFrame())

    # Defaults: median for numeric, first category for categorical
    default_inputs: dict = {}
    for col in all_feats:
        if col in num_cols and col in feat_ranges:
            default_inputs[col] = feat_ranges[col]["median"]
        elif col in cat_cols and col in feat_options:
            opts = feat_options[col]
            default_inputs[col] = opts[0] if opts else "unknown"

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    st.subheader("🎛️ Accident Input Parameters")
    st.caption(
        "All fields are pre-filled with dataset medians/modes. "
        "Adjust the values that match your accident scenario."
    )

    # We'll expose the most informative features prominently
    # (weather, road type, speed, vehicles, time, location type, safety)
    PRIMARY_FEATURES = [
        "weather_condition_first",
        "road_surface_condition_first",
        "route_category_first",
        "congestion_level_first",
        "visibility_level_first",
        "primary_cause_first",
        "dominant_vehicle_type_first",
        "vehicle_count_first",
        "total_occupants_in_crash_first",
        "mean_speed_at_impact_kmph",
        "hour_of_day_first",
        "alcohol_suspected_flag_first",
        "road_hazard_flag_first",
        "helmet_seatbelt_usage_rate",
        "valid_license_ratio",
        "lane_count_first",
        "temp_celsius_first",
    ]
    # Filter to those actually present in the model
    primary_feats = [f for f in PRIMARY_FEATURES if f in all_feats]
    # Remaining features filled by defaults
    input_dict = dict(default_inputs)  # start with all defaults

    with st.form("prediction_form"):
        # ── Section 1: Environment & Road ────────────────────────────────────
        st.markdown("#### 🌦️ Environment & Road Conditions")
        env_cols = st.columns(3)

        env_field_map = {
            "weather_condition_first":       ("Weather Condition",       0),
            "road_surface_condition_first":  ("Road Surface Condition",  1),
            "visibility_level_first":        ("Visibility Level",        2),
            "congestion_level_first":        ("Congestion Level",        0),
            "route_category_first":          ("Road Type / Route",       1),
            "lane_count_first":              ("Number of Lanes",         2),
            "temp_celsius_first":            ("Temperature (°C)",        0),
            "road_hazard_flag_first":        ("Road Hazard Present?",    1),
            "signal_present_flag_first":     ("Traffic Signal Present?", 2),
        }

        for feat, (label, col_idx) in env_field_map.items():
            if feat not in all_feats:
                continue
            with env_cols[col_idx]:
                if feat in cat_cols:
                    opts = feat_options.get(feat, ["unknown"])
                    val  = st.selectbox(label, options=opts, key=f"form_{feat}")
                elif feat in num_cols:
                    meta = feat_ranges.get(feat, {})
                    mn   = float(meta.get("min",    0))
                    mx   = float(meta.get("max",  100))
                    med  = float(meta.get("median", 0))
                    # Boolean-like integer columns → radio
                    if mx <= 1 and mn >= 0:
                        val = st.radio(label, [0, 1], index=int(med), horizontal=True, key=f"form_{feat}")
                    else:
                        val = st.number_input(label, min_value=mn, max_value=mx,
                                              value=med, key=f"form_{feat}")
                else:
                    val = default_inputs.get(feat)
                input_dict[feat] = val

        st.markdown("#### 🚗 Vehicle & Crash Details")
        veh_cols = st.columns(3)

        veh_field_map = {
            "dominant_vehicle_type_first":   ("Primary Vehicle Type",       0),
            "vehicle_count_first":           ("Number of Vehicles",         1),
            "total_occupants_in_crash_first":("Total Occupants Involved",   2),
            "occupants_per_vehicle_first":   ("Occupants per Vehicle",      0),
            "mean_speed_at_impact_kmph":     ("Avg Speed at Impact (km/h)", 1),
            "alcohol_suspected_flag_first":  ("Alcohol Suspected?",         2),
            "primary_cause_first":           ("Primary Cause",              0),
        }

        for feat, (label, col_idx) in veh_field_map.items():
            if feat not in all_feats:
                continue
            with veh_cols[col_idx]:
                if feat in cat_cols:
                    opts = feat_options.get(feat, ["unknown"])
                    val  = st.selectbox(label, options=opts, key=f"form_{feat}")
                elif feat in num_cols:
                    meta = feat_ranges.get(feat, {})
                    mn   = float(meta.get("min",    0))
                    mx   = float(meta.get("max",  200))
                    med  = float(meta.get("median", 0))
                    if mx <= 1 and mn >= 0:
                        val = st.radio(label, [0, 1], index=int(med), horizontal=True, key=f"form_{feat}")
                    else:
                        step = max((mx - mn) / 200, 0.1)
                        val  = st.number_input(label, min_value=mn, max_value=mx,
                                               value=med, step=round(step, 2),
                                               key=f"form_{feat}")
                else:
                    val = default_inputs.get(feat)
                input_dict[feat] = val

        st.markdown("#### 👤 Occupant Safety & Demographics")
        saf_cols = st.columns(3)

        saf_field_map = {
            "helmet_seatbelt_usage_rate": ("Safety Gear Usage Rate (0–1)", 0),
            "valid_license_ratio":        ("Valid Licence Ratio (0–1)",    1),
            "male_ratio":                 ("Male Occupant Ratio (0–1)",    2),
            "mean_passenger_age":         ("Mean Passenger Age",           0),
            "airbag_deployed_flag_mean":  ("Airbag Deploy Rate (0–1)",     1),
            "hour_of_day_first":          ("Hour of Day (0–23)",           2),
        }

        for feat, (label, col_idx) in saf_field_map.items():
            if feat not in all_feats:
                continue
            with saf_cols[col_idx]:
                if feat in num_cols:
                    meta = feat_ranges.get(feat, {})
                    mn   = float(meta.get("min",  0))
                    mx   = float(meta.get("max",  1))
                    med  = float(meta.get("median", 0))
                    step = max((mx - mn) / 200, 0.01)
                    val  = st.number_input(label, min_value=mn, max_value=mx,
                                           value=med, step=round(step, 4),
                                           key=f"form_{feat}")
                else:
                    val = default_inputs.get(feat)
                input_dict[feat] = val

        st.markdown("---")
        submitted = st.form_submit_button("🔮  Predict Severity", type="primary", use_container_width=True)

    # ── PREDICTION RESULT ─────────────────────────────────────────────────────
    if submitted:
        with st.spinner("Running prediction..."):
            try:
                pred_label, probas, class_names = predict_severity(bundle, input_dict)
            except Exception as exc:
                import traceback
                st.error(f"Prediction failed: {exc}")
                st.code(traceback.format_exc())
                st.stop()

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        res_l, res_r = st.columns([1, 2])

        with res_l:
            st.markdown("**Predicted Severity:**")
            st.markdown(severity_badge(pred_label), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            max_prob   = float(probas.max())
            conf_label = "🟢 High" if max_prob > 0.65 else ("🟡 Medium" if max_prob > 0.45 else "🔴 Low")
            st.metric("Model Confidence", f"{max_prob:.1%}", conf_label)

            # Warning box for fatal
            if pred_label == "fatal":
                st.error(
                    "⚠️ **HIGH RISK** — The model predicts a **Fatal** outcome under "
                    "these conditions. Extreme caution and emergency preparedness advised."
                )
            elif pred_label == "major":
                st.warning(
                    "🟠 **ELEVATED RISK** — **Major** severity predicted. "
                    "Significant injuries likely."
                )
            else:
                st.success(
                    "🟢 **LOWER RISK** — **Minor** severity predicted. "
                    "Standard safety precautions should suffice."
                )

        with res_r:
            st.markdown("**Probability Distribution Across Severity Classes:**")
            proba_df = pd.DataFrame({
                "Class": list(class_names),
                "Probability": list(probas),
            })
            fig_p = px.bar(
                proba_df, x="Class", y="Probability", color="Class",
                color_discrete_map=SEVERITY_COLORS,
                text=[f"{p:.1%}" for p in probas],
                height=300,
            )
            fig_p.update_traces(textposition="outside")
            fig_p.update_layout(
                showlegend=False,
                yaxis=dict(range=[0, 1.2], tickformat=".0%"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # Feature importance global reference
        if not feat_imp.empty:
            st.markdown("---")
            st.markdown("### 🔍 Top Feature Drivers (Global Importance)")
            top_fi = feat_imp.head(12).sort_values("Importance")
            fig_fi = px.bar(
                top_fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                text=top_fi["Importance"].round(4), height=360,
            )
            fig_fi.update_traces(textposition="outside")
            fig_fi.update_layout(
                coloraxis_showscale=False, margin=dict(t=10, b=10, r=60)
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 4 — INDIA ACCIDENT MAP
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
elif page == "🗺️  India Accident Map":

    st.title("🗺️ India Accident Map")
    st.markdown(
        "Interactive geographic map of road accidents across India. "
        "Points are **colour-coded by severity** — zoom, pan, and click "
        "markers for details."
    )
    st.markdown("---")

    # ── DATA CHECK ────────────────────────────────────────────────────────────
    if st.session_state.df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`.")
        st.stop()

    df = st.session_state.df

    # Require lat/lon columns
    if "lat_coord_first" not in df.columns or "lon_coord_first" not in df.columns:
        st.error("Latitude/longitude columns not found in dataset.")
        st.stop()

    # Drop rows with missing coordinates
    map_df = df.dropna(subset=["lat_coord_first", "lon_coord_first"]).copy()
    # Restrict to plausible Indian geography
    map_df = map_df[
        (map_df["lat_coord_first"].between(6, 37)) &
        (map_df["lon_coord_first"].between(68, 98))
    ]

    # ── FILTER PANEL ─────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=SEVERITY_ORDER,
            default=SEVERITY_ORDER,
            help="Select which severity levels to display on the map.",
        )

    with col_f2:
        if "state_name_first" in map_df.columns:
            states = sorted(map_df["state_name_first"].dropna().unique().tolist())
            state_filter = st.multiselect(
                "Filter by State",
                options=states,
                default=[],
                placeholder="All states (default)",
                help="Leave blank to show all states.",
            )
        else:
            state_filter = []

    with col_f3:
        max_pts = st.slider(
            "Max points to plot",
            min_value=200, max_value=5000, value=1500, step=100,
            help="Limit the number of markers for browser performance.",
        )

    # Apply filters
    if severity_filter:
        map_df = map_df[map_df[TARGET_COL].isin(severity_filter)]
    if state_filter:
        map_df = map_df[map_df["state_name_first"].isin(state_filter)]

    # Sample for performance
    if len(map_df) > max_pts:
        map_df = map_df.sample(n=max_pts, random_state=42)

    st.markdown("---")

    # ── MAP SUMMARY METRICS ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📍 Points Shown",  f"{len(map_df):,}")
    if TARGET_COL in map_df.columns:
        m2.metric("🔴 Fatal",  f"{(map_df[TARGET_COL]=='fatal').sum():,}")
        m3.metric("🟠 Major",  f"{(map_df[TARGET_COL]=='major').sum():,}")
        m4.metric("🟢 Minor",  f"{(map_df[TARGET_COL]=='minor').sum():,}")

    st.markdown("---")

    # ── LEGEND ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        **Map Legend:**  
        🟢 **Minor** — Lower severity injuries &nbsp;&nbsp;
        🟠 **Major** — Significant injuries &nbsp;&nbsp;
        🔴 **Fatal** — Life-threatening / death
        """,
        unsafe_allow_html=False,
    )

    # ── BUILD FOLIUM MAP ──────────────────────────────────────────────────────
    if map_df.empty:
        st.warning("No data points match the current filters.")
    else:
        # Centre on India's geographic midpoint
        m = folium.Map(
            location=[20.5937, 78.9629],
            zoom_start=5,
            tiles="CartoDB positron",
        )

        # Add a cluster layer for better performance
        from folium.plugins import MarkerCluster
        cluster = MarkerCluster(
            options={"maxClusterRadius": 40, "disableClusteringAtZoom": 10}
        ).add_to(m)

        # Plot each accident as a CircleMarker
        for _, row in map_df.iterrows():
            sev   = str(row.get(TARGET_COL, "minor")).lower()
            color = FOLIUM_COLORS.get(sev, "blue")

            # Build popup HTML
            city  = row.get("city_name_first",  "N/A")
            state = row.get("state_name_first", "N/A")
            cause = row.get("primary_cause_first", "N/A")
            speed = row.get("mean_speed_at_impact_kmph", "N/A")
            popup_html = (
                f"<b>Severity:</b> {sev.upper()}<br>"
                f"<b>City:</b> {city}<br>"
                f"<b>State:</b> {state}<br>"
                f"<b>Cause:</b> {cause}<br>"
                f"<b>Speed:</b> {speed} km/h"
            )

            folium.CircleMarker(
                location=[row["lat_coord_first"], row["lon_coord_first"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                tooltip=f"{sev.upper()} — {city}, {state}",
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(cluster)

        # Render the map in Streamlit
        st_folium(m, width=None, height=550, returned_objects=[])

        st.markdown("---")

        # ── SUPPLEMENTARY: State-level chart ─────────────────────────────────
        if "state_name_first" in map_df.columns and TARGET_COL in map_df.columns:
            st.subheader("📊 Accident Counts by State")

            state_tab1, state_tab2 = st.tabs(["📊  Grouped Bar", "📋  Data Table"])

            with state_tab1:
                state_data = (
                    map_df.groupby(["state_name_first", TARGET_COL])
                    .size()
                    .reset_index(name="Count")
                )
                state_data.columns = ["State", "Severity", "Count"]
                fig_s = px.bar(
                    state_data, x="State", y="Count", color="Severity",
                    color_discrete_map=SEVERITY_COLORS,
                    category_orders={"Severity": SEVERITY_ORDER},
                    barmode="group", height=420,
                )
                fig_s.update_layout(xaxis_tickangle=35, margin=dict(t=10, b=80))
                st.plotly_chart(fig_s, use_container_width=True)

            with state_tab2:
                pivot = (
                    map_df.groupby(["state_name_first", TARGET_COL])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
                )
                pivot.columns.name = None
                pivot = pivot.rename(columns={"state_name_first": "State"})
                pivot["Total"] = pivot.drop(columns="State").sum(axis=1)
                pivot = pivot.sort_values("Total", ascending=False).reset_index(drop=True)
                st.dataframe(pivot, use_container_width=True, hide_index=True)


# ── END OF APP ────────────────────────────────────────────────────────────────
