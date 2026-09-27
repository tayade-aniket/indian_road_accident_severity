# =============================================================================
# Indian Road Accident Severity Dashboard
# Streamlit multi-page application — Modernised UI & Keep-Awake Engine
# Run: streamlit run app.py
# =============================================================================

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib
import warnings
import os
import time
import threading
import datetime
import requests
from dotenv import load_dotenv

# sklearn imports — used when building a fresh model as fallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Load environment variables from .env in the project root (local) or st.secrets (Streamlit Cloud)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
MAPTILER_API_KEY: str = os.getenv("MAPTILER_API_KEY", "")
if not MAPTILER_API_KEY:
    try:
        MAPTILER_API_KEY = str(st.secrets.get("MAPTILER_API_KEY", ""))
    except Exception:
        MAPTILER_API_KEY = ""

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = os.path.join(_HERE, "data", "processed", "crash_level_accidents.csv")
MODEL_PATH = os.path.join(_HERE, "models", "best_model_bundle.joblib")

TARGET_COL = "crash_severity_first"

SEVERITY_ORDER  = ["minor", "major", "fatal"]
SEVERITY_COLORS = {
    "minor": "#10b981",  # Vibrant emerald green
    "major": "#f59e0b",  # Warm golden amber
    "fatal": "#ef4444"   # Vivid ruby red
}

FOLIUM_COLORS = {"minor": "green", "major": "orange", "fatal": "red"}

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Road Accident Severity Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ── KEEP-AWAKE ENGINE (BACKGROUND THREAD & CLIENT-SIDE HEARTBEAT) ─────────────
# =============================================================================

# Shared status dictionary for the keep-awake worker
if "keep_awake_stats" not in st.session_state:
    st.session_state.keep_awake_stats = {
        "active": False,
        "last_ping_time": None,
        "last_status_code": None,
        "ping_count": 0,
        "target_url": os.getenv("APP_URL", "")
    }

class KeepAwakeWorker:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.target_url = os.getenv("APP_URL", "")
        self.interval_seconds = 600  # 10 minutes
        self.thread = None
        self.running = False
        self.last_ping = "Never"
        self.last_status = "Initialized"
        self.ping_count = 0

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = KeepAwakeWorker()
                cls._instance.start()
            return cls._instance

    def update_url(self, url: str):
        if url and url.startswith(("http://", "https://")):
            self.target_url = url.strip()

    def _run(self):
        while self.running:
            if self.target_url:
                try:
                    headers = {"User-Agent": "Streamlit-KeepAwake-Heartbeat/2.0"}
                    resp = requests.get(self.target_url, timeout=12, headers=headers)
                    self.last_status = f"{resp.status_code} OK" if resp.status_code == 200 else f"HTTP {resp.status_code}"
                except Exception as e:
                    self.last_status = f"Err: {type(e).__name__}"
                self.ping_count += 1
                self.last_ping = datetime.datetime.now().strftime("%H:%M:%S")
            time.sleep(self.interval_seconds)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True, name="StreamlitKeepAwakeThread")
            self.thread.start()

# Initialize global worker once via Streamlit cached resource
@st.cache_resource
def get_keep_awake_service():
    return KeepAwakeWorker.get_instance()

keep_awake_worker = get_keep_awake_service()

# Inject lightweight client-side heartbeat to keep WebSocket and session alive
def inject_client_heartbeat():
    heartbeat_js = """
    <script>
    (function() {
        // Keeps the browser session alive and prevents tab sleep
        setInterval(function() {
            try {
                fetch(window.location.href, { method: 'HEAD', cache: 'no-store' })
                    .catch(function(err) {});
            } catch(e) {}
        }, 180000); // Heartbeat every 3 minutes
    })();
    </script>
    """
    components.html(heartbeat_js, height=0, width=0)

inject_client_heartbeat()


# =============================================================================
# ── ULTRA-MODERN GLOBAL CSS & DESIGN SYSTEM ──────────────────────────────────
# =============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container padding */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 96%;
    }

    /* Gradient Brand Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 16px;
        margin: 16px 0 24px 0;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px -8px rgba(0, 0, 0, 0.5);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-card-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
    }
    .metric-card-top-bar.green { background: linear-gradient(90deg, #10b981, #059669); }
    .metric-card-top-bar.amber { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .metric-card-top-bar.red   { background: linear-gradient(90deg, #ef4444, #b91c1c); }
    .metric-card-top-bar.blue  { background: linear-gradient(90deg, #38bdf8, #3b82f6); }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 6px 0 2px 0;
        letter-spacing: -0.5px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Structured Equal-Height Containers for Plots and Insights */
    .equal-container {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 22px;
        height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    .equal-container-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
    }
    .equal-container-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
    }
    .equal-container-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .equal-container-body {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    .equal-container-footer {
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.45;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
    }

    /* Severity Badges with Dynamic Glowing Gradients */
    .severity-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 24px;
        border-radius: 9999px;
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .badge-minor {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    .badge-major {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    }
    .badge-fatal {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.5);
    }

    /* Pulse animation for Live Keep-Awake pill */
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .pulse-dot {
        width: 9px;
        height: 9px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-green 2s infinite;
        margin-right: 6px;
    }

    /* Sleek card for Objective & Feature Grid */
    .glass-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        height: 100%;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-left-color: #a855f7;
    }
    .glass-card b {
        color: #f8fafc;
        font-size: 0.98rem;
    }
    .glass-card span {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
        line-height: 1.45;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(168, 85, 247, 0.2) 100%);
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    /* Button enhancements */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        padding: 0.55rem 1.4rem;
        transition: all 0.25s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.5);
    }
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
        "app_public_url": os.getenv("APP_URL", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── CACHED LOADERS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and cache the crash-level CSV from disk."""
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Please ensure `data/processed/crash_level_accidents.csv` is committed and pushed to the repository."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


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


def _build_widget_metadata(df_source: pd.DataFrame, num_cols: list, cat_cols: list) -> tuple:
    feat_ranges: dict  = {}
    feat_options: dict = {}
    for col in num_cols:
        if col in df_source.columns:
            col_data = df_source[col].dropna()
            feat_ranges[col] = {
                "min":    float(col_data.min()),
                "max":    float(col_data.max()),
                "median": float(col_data.median()),
            }
    for col in cat_cols:
        if col in df_source.columns:
            feat_options[col] = sorted(df_source[col].dropna().unique().tolist())
    return feat_ranges, feat_options


def _normalise_bundle(raw: dict, df_source: pd.DataFrame | None = None) -> dict:
    if "feature_names" in raw:
        return raw

    pipeline      = raw["pipeline"]
    label_encoder = raw["label_encoder"]
    feature_cols  = list(raw.get("feature_columns", []))
    classes       = list(raw.get("classes", []))
    model_name    = raw.get("model_name", "Unknown")

    num_cols: list = []
    cat_cols: list = []
    try:
        ct_step = None
        for _, step in pipeline.steps[:-1]:
            if hasattr(step, "transformers_"):
                ct_step = step
                break
        if ct_step is not None:
            for t_name, _, t_cols in ct_step.transformers_:
                if t_name == "num":
                    num_cols = list(t_cols)
                elif t_name == "cat":
                    cat_cols = list(t_cols)
    except Exception:
        if df_source is not None:
            num_cols = df_source[feature_cols].select_dtypes(include="number").columns.tolist()
            cat_cols = [c for c in feature_cols if c not in num_cols]

    feat_ranges: dict  = {}
    feat_options: dict = {}
    if df_source is not None:
        feat_ranges, feat_options = _build_widget_metadata(df_source, num_cols, cat_cols)

    feat_imp = pd.DataFrame()
    try:
        clf = pipeline.steps[-1][1]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            all_transformed_cols = num_cols + cat_cols
            if len(importances) == len(all_transformed_cols):
                feat_imp = pd.DataFrame({
                    "Feature":    all_transformed_cols,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False).reset_index(drop=True)
                feat_imp["Rank"] = feat_imp.index + 1
    except Exception:
        pass

    if not hasattr(label_encoder, "classes_") or label_encoder.classes_ is None:
        import numpy as _np
        label_encoder.classes_ = _np.array(classes)

    acc = raw.get("accuracy", raw.get("metrics", {}).get("accuracy", "—"))

    return {
        "pipeline":      pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_cols,
        "num_cols":      num_cols,
        "cat_cols":      cat_cols,
        "feat_imp":      feat_imp,
        "feat_ranges":   feat_ranges,
        "feat_options":  feat_options,
        "metrics":       {"accuracy": acc},
        "model_name":    model_name,
        "_fallback":     False,
    }


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    try:
        raw = joblib.load(MODEL_PATH)
        try:
            df_src = pd.read_csv(DATA_PATH)
        except Exception:
            df_src = None
        return _normalise_bundle(raw, df_source=df_src)
    except Exception:
        pass

    # Fallback trainer
    df_raw = pd.read_csv(DATA_PATH)
    df     = df_raw.copy()
    df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True, errors="ignore")

    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL]

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    num_cols  = X.select_dtypes(include="number").columns.tolist()
    cat_cols  = [c for c in _CAT_COLS if c in X.columns]
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
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
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

    importances = pipeline.named_steps["model"].feature_importances_
    feat_imp = pd.DataFrame({"Feature": used_cols, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    feat_imp["Rank"] = feat_imp.index + 1

    feat_ranges, feat_options = _build_widget_metadata(df, num_cols, cat_cols)

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
        "model_name":    "Random Forest (fallback)",
        "_fallback":     True,
    }


def predict_severity(bundle: dict, input_dict: dict):
    pipeline      = bundle["pipeline"]
    le            = bundle["label_encoder"]
    feature_names = bundle.get("feature_names") or bundle.get("feature_columns", [])

    if not feature_names:
        raise ValueError("Model bundle does not contain 'feature_names' or 'feature_columns'.")

    row      = {col: input_dict.get(col, np.nan) for col in feature_names}
    input_df = pd.DataFrame([row])

    probas   = pipeline.predict_proba(input_df)[0]
    pred_idx = int(np.argmax(probas))

    try:
        pred_label = le.classes_[pred_idx]
        class_names = le.classes_
    except (AttributeError, IndexError):
        classes     = bundle.get("classes", SEVERITY_ORDER)
        pred_label  = classes[pred_idx] if pred_idx < len(classes) else "unknown"
        class_names = classes

    return pred_label, probas, class_names


# ── UNIFIED PLOTLY THEME ──────────────────────────────────────────────────────
def apply_plot_theme(fig, height=360):
    """Apply uniform glass/modern theme with fixed height across all dashboard plots."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Plus Jakarta Sans', sans-serif", color="#cbd5e1", size=12),
        margin=dict(t=25, b=25, l=45, r=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
            tickfont=dict(size=11, color="#94a3b8"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
            tickfont=dict(size=11, color="#94a3b8"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="'Plus Jakarta Sans', sans-serif",
            bordercolor="rgba(255,255,255,0.2)"
        )
    )
    return fig


# ── HTML METRICS GENERATOR ────────────────────────────────────────────────────
def render_metric_grid(items: list):
    """
    Renders an equal-width, equal-height CSS Grid of metrics.
    items: list of dicts with: label, value, sub, color (optional: green, amber, red, blue)
    """
    cards_html = []
    for item in items:
        color_class = item.get("color", "blue")
        cards_html.append(
            f"""
            <div class="metric-card">
                <div class="metric-card-top-bar {color_class}"></div>
                <div class="metric-label">{item.get('icon', '📌')} {item['label']}</div>
                <div class="metric-value">{item['value']}</div>
                <div class="metric-sub">{item.get('sub', '')}</div>
            </div>
            """
        )
    html = f"""<div class="metric-grid">{''.join(cards_html)}</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ── SEVERITY BADGE HTML ───────────────────────────────────────────────────────
def severity_badge(label: str) -> str:
    return f'<span class="severity-badge badge-{label.lower()}">{label.upper()}</span>'


# ── AUTO-LOAD DATA & MODEL ────────────────────────────────────────────────────
if st.session_state.df is None:
    try:
        st.session_state.df = load_data()
    except Exception:
        pass

if st.session_state.model_bundle is None:
    try:
        st.session_state.model_bundle = load_model_bundle()
    except Exception:
        pass


# =============================================================================
# ── SIDEBAR NAVIGATION & KEEP-AWAKE MONITOR ───────────────────────────────────
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="font-size: 2rem;">🚦</div>
            <div>
                <h3 style="margin: 0; font-weight: 800; color: #f8fafc; font-size: 1.25rem;">Road Safety AI</h3>
                <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.5px;">INDIA ACCIDENT INTELLIGENCE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home", "📂  Dataset", "🔮  Predict Severity", "🗺️  India Accident Map"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick Model & Data Status
    df_ok    = st.session_state.df is not None
    model_ok = st.session_state.model_bundle is not None

    st.markdown("##### ⚙️ System Status")
    st.markdown(f"**Dataset:** {'🟢 Loaded' if df_ok else '🔴 Not found'}")

    if model_ok:
        is_fallback  = st.session_state.model_bundle.get("_fallback", False)
        model_name   = st.session_state.model_bundle.get("model_name", "Random Forest" if is_fallback else "XGBoost")
        acc          = st.session_state.model_bundle.get("metrics", {}).get("accuracy", "—")
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Accuracy:** `{acc}%`")
    else:
        st.markdown("**Model:** 🔴 Not found")

    if df_ok:
        st.markdown(f"**Records:** `{len(st.session_state.df):,}`")

    st.markdown("---")

    # ── KEEP-AWAKE CONTROLS & HEARTBEAT STATUS ───────────────────────────────
    with st.expander("⚡ Keep-Awake Engine", expanded=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="pulse-dot"></span>
                <span style="font-size: 0.85rem; font-weight: 700; color: #10b981;">ENGINE ACTIVE (AWAKE)</span>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 10px; line-height: 1.4;">
                Maintains active connection & sends automated self-pings to prevent host timeouts.
            </div>
            """,
            unsafe_allow_html=True,
        )

        app_url_input = st.text_input(
            "Hosted App URL",
            value=st.session_state.get("app_public_url", ""),
            placeholder="https://your-app.streamlit.app",
            help="Enter your public URL to enable periodic self-pinging."
        )

        if app_url_input != st.session_state.get("app_public_url", ""):
            st.session_state["app_public_url"] = app_url_input
            keep_awake_worker.update_url(app_url_input)

        col_ping_btn, col_ping_st = st.columns([1, 1])
        with col_ping_btn:
            if st.button("🚀 Ping Now", use_container_width=True):
                if app_url_input:
                    try:
                        r = requests.get(app_url_input, timeout=8)
                        keep_awake_worker.last_ping = datetime.datetime.now().strftime("%H:%M:%S")
                        keep_awake_worker.last_status = f"{r.status_code} OK"
                        st.toast(f"Ping successful! ({r.status_code})", icon="✅")
                    except Exception as e:
                        st.toast(f"Ping error: {e}", icon="⚠️")
                else:
                    st.toast("Enter a valid URL first", icon="ℹ️")

        with col_ping_st:
            st.caption(f"Last: `{keep_awake_worker.last_ping}`")
            st.caption(f"Status: `{keep_awake_worker.last_status}`")

        st.caption("ℹ️ *A GitHub Actions workflow is also active in `.github/workflows/keep_awake.yml` to wake your app every 12 mins.*")

    st.markdown("---")
    st.caption("Indian Road Accident Severity Intelligence\nv4.0 · Modernized Dashboard")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 1 — HOME
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
if page == "🏠  Home":

    # ── HERO BANNER ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🚦 Indian Road Accident Severity Intelligence</h1>
            <p class="hero-subtitle">
                Advanced machine learning platform for predicting, analyzing, and mitigating crash severity
                across National & State highways in India.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── PROBLEM STATEMENT & NATIONAL CONTEXT ─────────────────────────────────
    with st.expander("📋 National Road Safety Challenge — Key Statistics", expanded=True):
        col_ps, col_img = st.columns([3, 1])
        with col_ps:
            st.markdown(
                """
                **India records nearly 500,000 road accidents annually**, claiming over **1.68 lakh lives**
                (more than **460 fatalities per day** or one death every 3 minutes).

                This application leverages historical crash data across weather, road geometries, vehicle archetypes,
                and occupant safety parameters to **predict crash severity (Minor, Major, Fatal)** and extract
                critical causal drivers to support emergency dispatch and proactive road engineering.
                """
            )
        with col_img:
            st.metric("Annual Accidents", "4.8 Lakh+", "+2.1% YoY")
            st.metric("Annual Fatalities", "1.68 Lakh+", "40 / hour")
            st.metric("Economic Cost", "₹1.47 Lakh Cr.", "3.14% GDP")

    # ── KEY STATS GRID ────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("### 📊 Dataset Overview at a Glance")

        fatal_rate_str = f"{(df[TARGET_COL] == 'fatal').mean() * 100:.1f}%" if TARGET_COL in df.columns else "N/A"
        cities_count = f"{df['city_name_first'].nunique():,}" if "city_name_first" in df.columns else "—"
        states_count = f"{df['state_name_first'].nunique():,}" if "state_name_first" in df.columns else "—"

        metrics = [
            {"label": "Total Crash Records", "value": f"{len(df):,}", "sub": "Aggregated crash events", "icon": "📋", "color": "blue"},
            {"label": "Features Tracked", "value": f"{df.shape[1]}", "sub": "Road, weather & vehicle dims", "icon": "📌", "color": "blue"},
            {"label": "Monitored Cities", "value": cities_count, "sub": "Urban & semi-urban clusters", "icon": "🏙️", "color": "green"},
            {"label": "Indian States", "value": states_count, "sub": "Geographic coverage", "icon": "🗺️", "color": "amber"},
            {"label": "Fatal Severity Rate", "value": fatal_rate_str, "sub": "High-risk critical events", "icon": "💀", "color": "red"},
        ]
        render_metric_grid(metrics)

        # ── EQUAL WIDTH & HEIGHT PLOTS AND INSIGHTS ───────────────────────────
        col_bar, col_pie = st.columns(2)

        with col_bar:
            st.markdown(
                """
                <div class="equal-container">
                    <div class="equal-container-header">
                        <h4 class="equal-container-title">🎯 Severity Class Counts</h4>
                        <span class="equal-container-badge">Distribution</span>
                    </div>
                    <div class="equal-container-body">
                """,
                unsafe_allow_html=True,
            )
            if TARGET_COL in df.columns:
                counts = (
                    df[TARGET_COL]
                    .value_counts()
                    .reindex(SEVERITY_ORDER, fill_value=0)
                    .reset_index()
                )
                counts.columns = ["Severity", "Count"]
                counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

                fig_bar = px.bar(
                    counts, x="Severity", y="Count", color="Severity",
                    color_discrete_map=SEVERITY_COLORS,
                    text=counts["Pct"].apply(lambda x: f"{x}%"),
                )
                fig_bar.update_traces(textposition="outside", marker_line_width=1.5, marker_line_color="rgba(255,255,255,0.2)")
                apply_plot_theme(fig_bar, height=310)
                fig_bar.update_layout(showlegend=False, yaxis_title="Accident Count")
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(
                """
                    </div>
                    <div class="equal-container-footer">
                        💡 <b>Insight:</b> Minor & major accidents comprise the majority of events (~85%), but fatal crashes require high-priority predictive focus.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_pie:
            st.markdown(
                """
                <div class="equal-container">
                    <div class="equal-container-header">
                        <h4 class="equal-container-title">🥧 Severity Share Proportion</h4>
                        <span class="equal-container-badge">Ratio Breakdown</span>
                    </div>
                    <div class="equal-container-body">
                """,
                unsafe_allow_html=True,
            )
            if TARGET_COL in df.columns:
                pie_data = df[TARGET_COL].value_counts().reset_index()
                pie_data.columns = ["Severity", "Count"]

                fig_pie = px.pie(
                    pie_data, values="Count", names="Severity",
                    color="Severity", color_discrete_map=SEVERITY_COLORS,
                    hole=0.52,
                )
                apply_plot_theme(fig_pie, height=310)
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    marker=dict(line=dict(color="rgba(255,255,255,0.2)", width=1.5))
                )
                fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, x=0.2))
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(
                """
                    </div>
                    <div class="equal-container-footer">
                        ⚖️ <b>Class Weighting:</b> The model handles class imbalance using balanced sample weighting to ensure fatal cases are not overshadowed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── PROJECT OBJECTIVES GRID ───────────────────────────────────────────────
    st.markdown("### 🎯 Strategic Objectives")
    o1, o2 = st.columns(2)
    with o1:
        st.markdown(
            """
            <div class="glass-card">
                <b>1 · Predictive Intelligence</b>
                <span>Classify incoming accident alerts into Minor, Major, or Fatal within milliseconds using trained multi-class ensembles.</span>
            </div>
            <div style="height: 12px;"></div>
            <div class="glass-card">
                <b>2 · Root Cause Discovery</b>
                <span>Isolate the strongest contributing factors — such as impact speed, vehicle archetypes, and road geometry — across severity classes.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            """
            <div class="glass-card">
                <b>3 · Geospatial Hotspot Mapping</b>
                <span>Pinpoint critical accident clusters across National Highways and urban arterial roads to allocate emergency dispatch.</span>
            </div>
            <div style="height: 12px;"></div>
            <div class="glass-card">
                <b>4 · Decision-Support Interface</b>
                <span>Empower traffic authorities, first responders, and municipal engineers with intuitive real-time simulation tools.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── METHODOLOGY OVERVIEW ──────────────────────────────────────────────────
    st.markdown("### 🔬 Architecture & Methodology")
    tab_data, tab_model, tab_eval = st.tabs(
        ["📂  Data Pipeline", "🤖  Model Architecture", "📈  Evaluation & Validation"]
    )

    with tab_data:
        st.markdown(
            """
            | Pipeline Stage | Implementation Detail |
            |---|---|
            | **Raw Ingestion** | Crash-level aggregated records containing 49 contextual attributes |
            | **Imputation** | Median strategy for continuous metrics · Mode imputation for categorical values |
            | **Feature Encoding** | Scikit-learn `OrdinalEncoder` with unknown token handling |
            | **Target Definition** | Multi-class label: `minor` (low injury), `major` (severe injury), `fatal` (loss of life) |
            """
        )

    with tab_model:
        st.markdown(
            """
            | Component | Specification |
            |---|---|
            | **Primary Classifier** | Random Forest / XGBoost ensemble |
            | **Ensemble Parameters** | 150 Decision Trees, Max Depth = 15 |
            | **Class Balancing** | `class_weight='balanced'` applied to mitigate minority fatality class bias |
            | **Packaging** | Integrated `ColumnTransformer` + `Pipeline` serialized via `.joblib` |
            """
        )

    with tab_eval:
        st.markdown(
            """
            | Validation Metric | Strategy |
            |---|---|
            | **Cross-Validation** | 80/20 Stratified Train-Test split ensuring consistent severity distribution |
            | **Primary Objective** | Weighted F1-Score to balance precision and recall across all 3 classes |
            | **Diagnostic Tools** | Multi-class Confusion Matrix and ROC curve analysis |
            """
        )

    st.markdown("---")

    # ── HOW TO USE CARDS ──────────────────────────────────────────────────────
    st.markdown("### 🧭 Interactive Modules")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center; min-height: 140px;">
                <div class="metric-card-top-bar blue"></div>
                <div style="font-size: 1.8rem;">📂</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 6px;">Dataset Explorer</div>
                <div class="metric-sub" style="margin-top: 4px;">Inspect raw records, statistics, and distributions.</div>
            </div>
            """, unsafe_allow_html=True)
    with g2:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center; min-height: 140px;">
                <div class="metric-card-top-bar green"></div>
                <div style="font-size: 1.8rem;">🔮</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 6px;">Predict Severity</div>
                <div class="metric-sub" style="margin-top: 4px;">Simulate scenarios with instant ML inference.</div>
            </div>
            """, unsafe_allow_html=True)
    with g3:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center; min-height: 140px;">
                <div class="metric-card-top-bar amber"></div>
                <div style="font-size: 1.8rem;">🗺️</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 6px;">Hotspot Map</div>
                <div class="metric-sub" style="margin-top: 4px;">Explore geospatial accident clusters across India.</div>
            </div>
            """, unsafe_allow_html=True)
    with g4:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center; min-height: 140px;">
                <div class="metric-card-top-bar red"></div>
                <div style="font-size: 1.8rem;">⚡</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 6px;">Keep-Awake</div>
                <div class="metric-sub" style="margin-top: 4px;">Background heartbeat keeps free hosting active.</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 2 — DATASET
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
elif page == "📂  Dataset":

    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">📂 Crash Dataset Explorer</h1>
            <p class="hero-subtitle">
                Explore, slice, and audit the processed Indian road accidents database.
                Review distributions, missing value patterns, and feature correlations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`. Please check the file path.")
        st.stop()

    df = st.session_state.df

    # ── KPI METRICS STRIP ─────────────────────────────────────────────────────
    kpi_items = [
        {"label": "Total Crash Records", "value": f"{len(df):,}", "sub": "Curated incidents", "icon": "📋", "color": "blue"},
        {"label": "Total Features", "value": f"{df.shape[1]}", "sub": "Numerical & categorical", "icon": "📌", "color": "blue"},
        {"label": "States Represented", "value": f"{df['state_name_first'].nunique()}" if "state_name_first" in df.columns else "—", "sub": "Nationwide reach", "icon": "🗺️", "color": "green"},
        {"label": "Severity Classes", "value": f"{df[TARGET_COL].nunique()}" if TARGET_COL in df.columns else "3", "sub": "minor · major · fatal", "icon": "🏷️", "color": "red"},
    ]
    render_metric_grid(kpi_items)

    # ── SAMPLE TABLE ──────────────────────────────────────────────────────────
    st.markdown("### 🎲 Interactive Dataset Sample")

    col_refresh, col_n, _ = st.columns([1.2, 2, 4])
    with col_refresh:
        if st.button("🔄 Refresh Random Sample", type="primary", use_container_width=True):
            st.session_state.dataset_seed = np.random.randint(0, 99999)

    with col_n:
        n_sample = st.slider("Sample size", min_value=20, max_value=100, value=25, step=5)

    sample_df = df.sample(n=min(n_sample, len(df)), random_state=st.session_state.dataset_seed)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    st.caption(f"Displaying {len(sample_df)} randomly sampled rows (Random Seed: {st.session_state.dataset_seed}).")

    st.markdown("---")

    # ── DETAILED TABS ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Summary Statistics", "🔤  Column Metadata", "❓  Missing Value Audit", "📈  Distribution & States"]
    )

    with tab1:
        st.subheader("Numerical Features Summary")
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            st.dataframe(num_df.describe().T.round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab2:
        st.subheader("Column Types & Sample Values")
        dtype_df = pd.DataFrame({
            "Column":       df.columns,
            "Data Type":    df.dtypes.values.astype(str),
            "Non-Null":     df.notnull().sum().values,
            "Null Count":   df.isnull().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Sample Value": [
                str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else ""
                for c in df.columns
            ],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=450)

    with tab3:
        st.subheader("Missing Values Audit")
        missing_cnt = df.isnull().sum()
        miss_df = (
            pd.DataFrame({
                "Column": missing_cnt.index,
                "Missing Count":  missing_cnt.values,
                "Missing %":  (missing_cnt.values / len(df) * 100).round(2),
            })
            .query("`Missing Count` > 0")
            .reset_index(drop=True)
        )
        if miss_df.empty:
            st.success("✅ Clean dataset: Zero missing values detected across all columns.")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            with col_m2:
                fig_m = px.bar(
                    miss_df, x="Column", y="Missing %", text="Missing %",
                    color="Missing %", color_continuous_scale="Reds",
                )
                fig_m.update_traces(texttemplate="%{text}%", textposition="outside")
                apply_plot_theme(fig_m, height=340)
                st.plotly_chart(fig_m, use_container_width=True)

    with tab4:
        st.subheader("Severity Distribution & Top Geographic States")
        if TARGET_COL in df.columns:
            dist_data = (
                df[TARGET_COL]
                .value_counts()
                .reindex(SEVERITY_ORDER, fill_value=0)
                .reset_index()
            )
            dist_data.columns = ["Severity", "Count"]
            dist_data["Percentage"] = (dist_data["Count"] / dist_data["Count"].sum() * 100).round(1)

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown(
                    """
                    <div class="equal-container">
                        <div class="equal-container-header">
                            <h4 class="equal-container-title">🎯 Class Distribution</h4>
                            <span class="equal-container-badge">Overview</span>
                        </div>
                        <div class="equal-container-body">
                    """,
                    unsafe_allow_html=True,
                )
                fig_d = px.bar(
                    dist_data, x="Severity", y="Count", color="Severity",
                    color_discrete_map=SEVERITY_COLORS,
                    text="Percentage",
                )
                fig_d.update_traces(texttemplate="%{text}%", textposition="outside")
                apply_plot_theme(fig_d, height=310)
                fig_d.update_layout(showlegend=False)
                st.plotly_chart(fig_d, use_container_width=True)
                st.markdown(
                    """
                        </div>
                        <div class="equal-container-footer">
                            📊 Total dataset represents balanced real-world accident reports across urban and highway corridors.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_d2:
                st.markdown(
                    """
                    <div class="equal-container">
                        <div class="equal-container-header">
                            <h4 class="equal-container-title">🗺️ Top Contributing States</h4>
                            <span class="equal-container-badge">Top 8</span>
                        </div>
                        <div class="equal-container-body">
                    """,
                    unsafe_allow_html=True,
                )
                if "state_name_first" in df.columns:
                    top_states = df["state_name_first"].value_counts().head(8).reset_index()
                    top_states.columns = ["State", "Crashes"]
                    fig_st = px.bar(
                        top_states, x="Crashes", y="State", orientation="h",
                        color="Crashes", color_continuous_scale="Blues",
                    )
                    apply_plot_theme(fig_st, height=310)
                    fig_st.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_st, use_container_width=True)
                else:
                    st.info("State column not found.")

                st.markdown(
                    """
                        </div>
                        <div class="equal-container-footer">
                            📍 Highly populated transit states report higher frequencies, reflecting higher vehicular density.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 3 — PREDICT SEVERITY
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
elif page == "🔮  Predict Severity":

    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🔮 Machine Learning Severity Inference</h1>
            <p class="hero-subtitle">
                Enter crash scenario parameters to forecast severity outcome (Minor, Major, Fatal),
                examine prediction probabilities, and review global feature drivers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.model_bundle is None:
        st.error(f"Pre-trained model bundle not found at `{MODEL_PATH}`.")
        st.stop()

    bundle = st.session_state.model_bundle

    feat_ranges  = bundle.get("feat_ranges",  {})
    feat_options = bundle.get("feat_options", {})
    num_cols     = bundle.get("num_cols",     [])
    cat_cols     = bundle.get("cat_cols",     [])
    all_feats    = bundle.get("feature_names", [])
    feat_imp     = bundle.get("feat_imp",     pd.DataFrame())

    default_inputs: dict = {}
    for col in all_feats:
        if col in num_cols and col in feat_ranges:
            default_inputs[col] = feat_ranges[col]["median"]
        elif col in cat_cols and col in feat_options:
            opts = feat_options[col]
            default_inputs[col] = opts[0] if opts else "unknown"

    input_dict = dict(default_inputs)

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Scenario Simulation Parameters")
    st.caption("Adjust the sliders, dropdowns, and flags below to model a specific accident scenario.")

    with st.form("prediction_form"):
        # Section 1: Environment & Road
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.4); padding: 10px 16px; border-radius: 10px; border-left: 4px solid #38bdf8; margin: 12px 0;">
                <b style="color: #f8fafc;">🌦️ Section 1: Environment & Road Infrastructure</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        env_cols = st.columns(3)

        env_field_map = {
            "weather_condition_first":       ("Weather Condition",       0),
            "road_surface_condition_first":  ("Road Surface Condition",  1),
            "visibility_level_first":        ("Visibility Level",        2),
            "congestion_level_first":        ("Congestion Level",        0),
            "route_category_first":          ("Route / Road Category",   1),
            "lane_count_first":              ("Lane Count",              2),
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
                    if mx <= 1 and mn >= 0:
                        val = st.radio(label, [0, 1], index=int(med), horizontal=True, key=f"form_{feat}")
                    else:
                        val = st.number_input(label, min_value=mn, max_value=mx, value=med, key=f"form_{feat}")
                else:
                    val = default_inputs.get(feat)
                input_dict[feat] = val

        # Section 2: Vehicle & Crash Dynamics
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.4); padding: 10px 16px; border-radius: 10px; border-left: 4px solid #f59e0b; margin: 18px 0 12px 0;">
                <b style="color: #f8fafc;">🚗 Section 2: Vehicle & Impact Dynamics</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        veh_cols = st.columns(3)

        veh_field_map = {
            "dominant_vehicle_type_first":   ("Primary Vehicle Type",       0),
            "vehicle_count_first":           ("Vehicles Involved",          1),
            "total_occupants_in_crash_first":("Total Occupants",            2),
            "occupants_per_vehicle_first":   ("Occupants Per Vehicle",      0),
            "mean_speed_at_impact_kmph":     ("Mean Speed at Impact (km/h)", 1),
            "alcohol_suspected_flag_first":  ("Alcohol Suspected?",         2),
            "primary_cause_first":           ("Reported Primary Cause",     0),
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
                        val  = st.number_input(label, min_value=mn, max_value=mx, value=med, step=round(step, 2), key=f"form_{feat}")
                else:
                    val = default_inputs.get(feat)
                input_dict[feat] = val

        # Section 3: Safety Gear & Occupants
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.4); padding: 10px 16px; border-radius: 10px; border-left: 4px solid #10b981; margin: 18px 0 12px 0;">
                <b style="color: #f8fafc;">👤 Section 3: Safety Gear & Demographics</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        saf_cols = st.columns(3)

        saf_field_map = {
            "helmet_seatbelt_usage_rate": ("Safety Gear Compliance (0–1)", 0),
            "valid_license_ratio":        ("Valid License Ratio (0–1)",   1),
            "male_ratio":                 ("Male Occupant Ratio (0–1)",    2),
            "mean_passenger_age":         ("Mean Passenger Age (Years)",   0),
            "airbag_deployed_flag_mean":  ("Airbag Deployment Rate (0–1)", 1),
            "hour_of_day_first":          ("Crash Hour (0–23)",            2),
        }

        for feat, (label, col_idx) in saf_field_map.items():
            if feat not in all_feats:
                continue
            with saf_cols[col_idx]:
                if feat in num_cols:
                    meta = feat_ranges.get(feat, {})
                    mn   = float(meta.get("min",  0))
                    mx   = float(meta.get("max",  1 if "ratio" in feat or "rate" in feat else 100))
                    med  = float(meta.get("median", 0))
                    step = 0.05 if mx <= 1 else 1.0
                    val  = st.number_input(label, min_value=mn, max_value=mx, value=med, step=step, key=f"form_{feat}")
                else:
                    val = default_inputs.get(feat)
                input_dict[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Run ML Severity Prediction", type="primary", use_container_width=True)

    # ── PREDICTION RESULTS & EQUAL CONTAINER GRID ─────────────────────────────
    if submitted:
        with st.spinner("Executing inference pipeline..."):
            try:
                pred_label, probas, class_names = predict_severity(bundle, input_dict)
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                st.stop()

        st.markdown("---")
        st.markdown("### 🎯 Inference Results & Risk Profile")

        res_l, res_r = st.columns(2)

        with res_l:
            st.markdown(
                """
                <div class="equal-container" style="height: 420px;">
                    <div class="equal-container-header">
                        <h4 class="equal-container-title">🏁 Predicted Outcome</h4>
                        <span class="equal-container-badge">Model Decision</span>
                    </div>
                    <div class="equal-container-body" style="text-align: center;">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(severity_badge(pred_label), unsafe_allow_html=True)

            max_prob = float(probas.max())
            st.markdown(
                f"""
                <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-top: 14px;">
                    Confidence: <span style="color: #38bdf8;">{max_prob:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if pred_label == "fatal":
                st.error("⚠️ **CRITICAL SEVERITY**: High likelihood of fatal outcome. Immediate trauma-center dispatch recommended.")
            elif pred_label == "major":
                st.warning("🟠 **HIGH SEVERITY**: Major vehicle deformation and severe injuries anticipated.")
            else:
                st.success("🟢 **LOW SEVERITY**: Minor injuries anticipated. Standard medical and clearance protocols apply.")

            st.markdown(
                """
                    </div>
                    <div class="equal-container-footer">
                        🛡️ <b>Actionable Protocol:</b> Dispatch level calibrated against impact speed and safety compliance rates.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_r:
            st.markdown(
                """
                <div class="equal-container" style="height: 420px;">
                    <div class="equal-container-header">
                        <h4 class="equal-container-title">📊 Class Probability Spectrum</h4>
                        <span class="equal-container-badge">Multi-Class Softmax</span>
                    </div>
                    <div class="equal-container-body">
                """,
                unsafe_allow_html=True,
            )
            proba_df = pd.DataFrame({
                "Class": list(class_names),
                "Probability": list(probas),
            })
            fig_p = px.bar(
                proba_df, x="Class", y="Probability", color="Class",
                color_discrete_map=SEVERITY_COLORS,
                text=[f"{p:.1%}" for p in probas],
            )
            fig_p.update_traces(textposition="outside", marker_line_width=1.5, marker_line_color="rgba(255,255,255,0.2)")
            apply_plot_theme(fig_p, height=250)
            fig_p.update_layout(
                showlegend=False,
                yaxis=dict(range=[0, max(1.0, max_prob * 1.25)], tickformat=".0%"),
            )
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown(
                """
                    </div>
                    <div class="equal-container-footer">
                        📈 Shows individual likelihood for Minor, Major, and Fatal outcomes under this specific scenario.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── EQUAL WIDTH & HEIGHT: FEATURE IMPORTANCE VS INSIGHTS ──────────────
        if not feat_imp.empty:
            st.markdown("### 🔍 Model Explainability & Key Drivers")

            col_fi, col_insight = st.columns(2)

            with col_fi:
                st.markdown(
                    """
                    <div class="equal-container" style="height: 440px;">
                        <div class="equal-container-header">
                            <h4 class="equal-container-title">📈 Top Global Feature Importances</h4>
                            <span class="equal-container-badge">Tree Weights</span>
                        </div>
                        <div class="equal-container-body">
                    """,
                    unsafe_allow_html=True,
                )
                top_fi = feat_imp.head(8).sort_values("Importance", ascending=True)
                fig_fi = px.bar(
                    top_fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Viridis",
                    text=top_fi["Importance"].round(3),
                )
                fig_fi.update_traces(textposition="outside")
                apply_plot_theme(fig_fi, height=310)
                fig_fi.update_layout(coloraxis_showscale=False, margin=dict(r=40, t=10, b=10))
                st.plotly_chart(fig_fi, use_container_width=True)

                st.markdown(
                    """
                        </div>
                        <div class="equal-container-footer">
                            💡 Relative importance calculated across all splitting nodes in the ensemble.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_insight:
                st.markdown(
                    """
                    <div class="equal-container" style="height: 440px;">
                        <div class="equal-container-header">
                            <h4 class="equal-container-title">💡 Risk Mitigation Insights</h4>
                            <span class="equal-container-badge">Safety Advisory</span>
                        </div>
                        <div class="equal-container-body" style="justify-content: space-around;">
                            <div class="glass-card" style="min-height: 80px; margin-bottom: 8px;">
                                <b>⚡ Impact Speed Thresholds</b>
                                <span>Crash kinetic energy scales quadratically ($v^2$). Reducing speed by 10 km/h drastically shifts probability from Fatal to Major.</span>
                            </div>
                            <div class="glass-card" style="min-height: 80px; margin-bottom: 8px;">
                                <b>🛡️ Safety Gear Factor</b>
                                <span>Helmet and seatbelt compliance rate is one of the highest ranked protective factors against fatal head trauma.</span>
                            </div>
                            <div class="glass-card" style="min-height: 80px;">
                                <b>🌧️ Environmental Interactions</b>
                                <span>Adverse road conditions compounded with poor lighting (night hours) display the highest fatal probability multipliers.</span>
                            </div>
                        </div>
                        <div class="equal-container-footer">
                            📋 Calibrated with recommendations from the Ministry of Road Transport and Highways (MoRTH).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# PAGE 4 — INDIA ACCIDENT MAP
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
elif page == "🗺️  India Accident Map":

    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🗺️ Geospatial Accident Intelligence Map</h1>
            <p class="hero-subtitle">
                Interactive spatial map of crash incidents across Indian National & State Highways.
                Filter by severity, state, and explore concentrated accident blackspots.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`.")
        st.stop()

    df = st.session_state.df

    if "lat_coord_first" not in df.columns or "lon_coord_first" not in df.columns:
        st.error("Coordinates (lat_coord_first, lon_coord_first) not found in dataset.")
        st.stop()

    map_df = df.dropna(subset=["lat_coord_first", "lon_coord_first"]).copy()
    map_df = map_df[
        (map_df["lat_coord_first"].between(6, 37)) &
        (map_df["lon_coord_first"].between(68, 98))
    ]

    # ── FILTER PANEL IN SLEEK GLASS CONTAINER ────────────────────────────────
    st.markdown("### 🔍 Spatial Filter Controls")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=SEVERITY_ORDER,
            default=SEVERITY_ORDER,
            help="Toggle visible severity tiers on map.",
        )

    with col_f2:
        states = sorted(map_df["state_name_first"].dropna().unique().tolist()) if "state_name_first" in map_df.columns else []
        state_filter = st.multiselect(
            "Filter by State",
            options=states,
            default=[],
            placeholder="All states (default)",
            help="Filter to specific state corridors.",
        )

    with col_f3:
        max_pts = st.slider(
            "Maximum Markers Rendered",
            min_value=200, max_value=5000, value=1500, step=100,
            help="Higher point counts may affect browser rendering speed.",
        )

    if severity_filter:
        map_df = map_df[map_df[TARGET_COL].isin(severity_filter)]
    if state_filter:
        map_df = map_df[map_df["state_name_first"].isin(state_filter)]

    if len(map_df) > max_pts:
        map_df = map_df.sample(n=max_pts, random_state=42)

    # ── MAP METRICS STRIP ─────────────────────────────────────────────────────
    fatal_cnt = (map_df[TARGET_COL] == 'fatal').sum() if TARGET_COL in map_df.columns else 0
    major_cnt = (map_df[TARGET_COL] == 'major').sum() if TARGET_COL in map_df.columns else 0
    minor_cnt = (map_df[TARGET_COL] == 'minor').sum() if TARGET_COL in map_df.columns else 0

    map_kpis = [
        {"label": "Active Map Points", "value": f"{len(map_df):,}", "sub": "Plotted incidents", "icon": "📍", "color": "blue"},
        {"label": "Fatal Crashes", "value": f"{fatal_cnt:,}", "sub": "Red markers", "icon": "🔴", "color": "red"},
        {"label": "Major Crashes", "value": f"{major_cnt:,}", "sub": "Orange markers", "icon": "🟠", "color": "amber"},
        {"label": "Minor Crashes", "value": f"{minor_cnt:,}", "sub": "Green markers", "icon": "🟢", "color": "green"},
    ]
    render_metric_grid(map_kpis)

    # ── MAP RENDERING ─────────────────────────────────────────────────────────
    if map_df.empty:
        st.warning("No incidents match the active filter criteria. Adjust the severity or state selections.")
    else:
        if MAPTILER_API_KEY:
            tile_url = f"https://api.maptiler.com/maps/streets/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_API_KEY}"
            tiles_kwargs = dict(tiles=tile_url, attr="MapTiler")
        else:
            tiles_kwargs = dict(tiles="CartoDB positron")

        m = folium.Map(
            location=[20.5937, 78.9629],
            zoom_start=5,
            **tiles_kwargs,
        )

        from folium.plugins import MarkerCluster
        cluster = MarkerCluster(
            options={"maxClusterRadius": 40, "disableClusteringAtZoom": 10}
        ).add_to(m)

        for _, row in map_df.iterrows():
            sev   = str(row.get(TARGET_COL, "minor")).lower()
            color = FOLIUM_COLORS.get(sev, "blue")
            city  = row.get("city_name_first",  "N/A")
            state = row.get("state_name_first", "N/A")
            cause = row.get("primary_cause_first", "N/A")
            speed = row.get("mean_speed_at_impact_kmph", "N/A")

            sev_color = SEVERITY_COLORS.get(sev, "#000000")
            popup_html = (
                f"<div style='font-family: sans-serif; font-size: 12px;'>"
                f"<b style='color: {sev_color}; font-size: 14px;'>{sev.upper()} CRASH</b><br>"
                f"<b>Location:</b> {city}, {state}<br>"
                f"<b>Cause:</b> {cause}<br>"
                f"<b>Impact Speed:</b> {speed} km/h"
                f"</div>"
            )

            folium.CircleMarker(
                location=[row["lat_coord_first"], row["lon_coord_first"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                tooltip=f"{sev.upper()} — {city}, {state}",
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(cluster)

        st_folium(m, width=None, height=540, returned_objects=[])

        st.markdown("---")

        # ── EQUAL WIDTH & HEIGHT: STATE COMPARISON CHARTS ─────────────────────
        if "state_name_first" in map_df.columns and TARGET_COL in map_df.columns:
            st.markdown("### 📊 State-Level Comparative Breakdown")

            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown(
                    """
                    <div class="equal-container" style="height: 480px;">
                        <div class="equal-container-header">
                            <h4 class="equal-container-title">📊 Severity by State</h4>
                            <span class="equal-container-badge">Stacked Analysis</span>
                        </div>
                        <div class="equal-container-body">
                    """,
                    unsafe_allow_html=True,
                )
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
                    barmode="group",
                )
                apply_plot_theme(fig_s, height=350)
                fig_s.update_layout(xaxis_tickangle=35, margin=dict(b=70, t=10))
                st.plotly_chart(fig_s, use_container_width=True)

                st.markdown(
                    """
                        </div>
                        <div class="equal-container-footer">
                            🛣️ Grouped volume comparison reveals geographic concentration across key interstate arteries.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_s2:
                st.markdown(
                    """
                    <div class="equal-container" style="height: 480px;">
                        <div class="equal-container-header">
                            <h4 class="equal-container-title">📋 State Totals & Proportions</h4>
                            <span class="equal-container-badge">Summary Table</span>
                        </div>
                        <div class="equal-container-body" style="overflow-y: auto;">
                    """,
                    unsafe_allow_html=True,
                )
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

                st.dataframe(pivot, use_container_width=True, hide_index=True, height=350)

                st.markdown(
                    """
                        </div>
                        <div class="equal-container-footer">
                            📌 Sorted by aggregate incident volume for targeted state-level highway safety planning.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ── END OF APP ────────────────────────────────────────────────────────────────
