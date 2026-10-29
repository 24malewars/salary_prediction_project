import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(APP_DIR, "job_salary_prediction_dataset.csv")
MODEL_FILE = os.path.join(APP_DIR, "salary_model.joblib")
METRICS_FILE = os.path.join(APP_DIR, "model_metrics.json")

# Salaries in the dataset are treated as USD and converted at a fixed,
# illustrative exchange rate for display in Indian Rupees.
DEFAULT_USD_TO_INR = 95.24

NUM_COLS = ["experience_years", "skills_count", "certifications"]
CAT_COLS = ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "salary"

# Modeling settings (follow Job_Salary_Statistical_Inference_Project.ipynb)
RANDOM_STATE = 42
MODEL_SAMPLE_SIZE = 50000   # reproducible sample used for model training
TEST_SIZE = 0.2
PIPELINE_VERSION = "notebook_v1"

EDU_ORDER = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
SIZE_ORDER = ["Startup", "Small", "Medium", "Large", "Enterprise"]
REMOTE_ORDER = ["No", "Hybrid", "Yes"]

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS (Clean White Background + Vibrant Colorful Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SalarySense — Job Salary Analytics & Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Base: Crisp White Theme */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        background-color: #FFFFFF !important;
    }

    /* Headings & Text Color Overrides for White Theme */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 800;
    }
    p, span, label, div {
        color: #334155;
    }

    /* Sidebar Clean Light Background */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1.5px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1E293B !important;
    }

    /* Top Hero Header: Vibrant Contrast Card */
    .hero-banner {
        background: linear-gradient(135deg, #0A0F1D 0%, #1E1B4B 35%, #0F3E48 70%, #064E3B 100%);
        border: 1.5px solid #0D9488;
        padding: 28px 34px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        margin-top: 8px;
        color: #CBD5E1 !important;
        line-height: 1.6;
    }

    /* Colorful Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        transition: transform 0.15s ease;
    }
    .badge-pill:hover {
        transform: scale(1.04);
    }
    .pill-cyan {
        background: rgba(14, 165, 233, 0.2);
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    .pill-emerald {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399 !important;
        border: 1px solid rgba(52, 211, 153, 0.5);
    }
    .pill-purple {
        background: rgba(139, 92, 246, 0.2);
        color: #C084FC !important;
        border: 1px solid rgba(168, 85, 247, 0.5);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24 !important;
        border: 1px solid rgba(251, 191, 36, 0.5);
    }
    .pill-rose {
        background: rgba(244, 63, 94, 0.2);
        color: #FB7185 !important;
        border: 1px solid rgba(251, 113, 133, 0.5);
    }

    /* Vibrant Golden-Teal Exchange Rate Notice for White Theme */
    .rate-banner {
        background: linear-gradient(90deg, #FEF3C7 0%, #ECFDF5 100%);
        border-left: 5px solid #D97706;
        border-top: 1px solid #FDE68A;
        border-right: 1px solid #A7F3D0;
        border-bottom: 1px solid #A7F3D0;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .rate-banner * {
        color: #1E293B !important;
    }

    /* Colorful Glassmorphic KPI Cards on White Background */
    .metric-card {
        border-radius: 16px;
        padding: 20px 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }

    .card-cyan {
        background: #F0F9FF;
        border: 1.5px solid #7DD3FC;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.08);
    }
    .card-cyan:hover {
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.2);
    }

    .card-emerald {
        background: #ECFDF5;
        border: 1.5px solid #6EE7B7;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08);
    }
    .card-emerald:hover {
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
    }

    .card-purple {
        background: #F5F3FF;
        border: 1.5px solid #C4B5FD;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.08);
    }
    .card-purple:hover {
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
    }

    .card-coral {
        background: #FFF1F2;
        border: 1.5px solid #FDA4AF;
        box-shadow: 0 4px 14px rgba(244, 63, 94, 0.08);
    }
    .card-coral:hover {
        box-shadow: 0 8px 24px rgba(244, 63, 94, 0.2);
    }

    .metric-icon-wrap {
        width: 50px;
        height: 50px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        flex-shrink: 0;
    }
    .icon-cyan { background: #E0F2FE; border: 1px solid #38BDF8; color: #0284C7; }
    .icon-emerald { background: #D1FAE5; border: 1px solid #34D399; color: #059669; }
    .icon-purple { background: #EDE9FE; border: 1px solid #A855F7; color: #7C3AED; }
    .icon-coral { background: #FFE4E6; border: 1px solid #FB7185; color: #E11D48; }

    .text-cyan { color: #0284C7 !important; }
    .text-emerald { color: #059669 !important; }
    .text-purple { color: #7C3AED !important; }
    .text-coral { color: #E11D48 !important; }

    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748B !important;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 3px;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.83rem;
        color: #475569 !important;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Prediction Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #0A0F1D 0%, #1E1B4B 40%, #064E3B 100%);
        border: 2px solid #2DD4BF;
        border-radius: 18px;
        padding: 30px 34px;
        color: white;
        margin-top: 24px;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.15);
    }
    .prediction-card * {
        color: white;
    }
    .prediction-value-gradient {
        font-size: 3.1rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38BDF8 0%, #34D399 50%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }

    /* Form Submit & Action Buttons */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #6366F1 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
    }

    /* Missing Artifact Callout Box */
    .info-callout {
        background: #FEF3C7;
        border-left: 5px solid #F59E0B;
        border: 1px solid #FCD34D;
        padding: 16px 20px;
        border-radius: 12px;
        color: #92400E;
        margin-bottom: 22px;
    }
    .info-callout * {
        color: #92400E !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Shared chart styling
AXIS_STYLE = dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"))

def style_fig(fig, height=340, margin=None):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=margin or dict(l=20, r=20, t=30, b=20),
        height=height,
        xaxis=AXIS_STYLE,
        yaxis=AXIS_STYLE,
    )
    return fig

# -----------------------------------------------------------------------------
# Formatting Helpers (Indian Numbering System & LPA)
# -----------------------------------------------------------------------------
def format_inr(number):
    """Format a number in the Indian numbering style (e.g. ₹12,50,000)."""
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0"
    n = int(round(number))
    is_neg = n < 0
    s = str(abs(n))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        pairs = []
        while len(remaining) > 2:
            pairs.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            pairs.append(remaining)
        formatted = ",".join(reversed(pairs)) + "," + last3
    return f"{'-' if is_neg else ''}₹{formatted}"

def format_lpa(number):
    """Format a number into Lakhs Per Annum (e.g. ₹12.50 LPA). 1 Lakh = 1,00,000 INR."""
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0.00 LPA"
    return f"₹{number / 100000.0:,.2f} LPA"

def format_monthly_inr(number):
    """Format monthly equivalent of an annual salary in INR."""
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0 / mo"
    return f"{format_inr(number / 12.0)} / mo"

# -----------------------------------------------------------------------------
# Data & Model Artifact Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Loads and cleans the Job Salary Prediction dataset.
    Adds salary_in_inr and salary_in_lpa using the configured exchange rate.
    """
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)

    # Data cleaning (notebook section 2): drop exact duplicates, coerce numerics,
    # fill gaps defensively, and remove invalid numeric values.
    df = df.drop_duplicates().copy()
    for col in NUM_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    for col in CAT_COLS:
        mode = df[col].mode(dropna=True)
        df[col] = df[col].astype("string").fillna(mode.iloc[0] if len(mode) else "Unknown").astype(str)
    df = df[
        (df["experience_years"] >= 0) & (df["skills_count"] >= 0) &
        (df["certifications"] >= 0) & (df[TARGET] > 0)
    ].reset_index(drop=True)

    df["salary_in_usd"] = df[TARGET].astype(float)
    df["salary_in_inr"] = df["salary_in_usd"] * DEFAULT_USD_TO_INR
    df["salary_in_lpa"] = df["salary_in_inr"] / 100000.0
    return df

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Loads the trained pipeline from salary_model.joblib."""
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            # Only reject models trained on a different feature set
            cols = getattr(model, "feature_names_in_", None)
            if cols is not None and set(cols) != set(FEATURES):
                return None
            return model
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_model_metrics():
    """Loads model evaluation metrics and metadata from model_metrics.json."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            required = ["best_model_name", "best_model_metrics", "model_comparison"]
            if not isinstance(data, dict) or any(k not in data for k in required):
                return None
            best = data["best_model_metrics"]
            if "Test R²" not in best or "Test MAE (LPA)" not in best:
                return None
            return data
        except Exception:
            return None
    return None

def train_model_locally():
    """
    Mirrors the notebook's modeling workflow: trains Linear, Ridge and Lasso
    regression pipelines on a reproducible 50,000-row sample (80/20 split), picks the
    best by held-out R² (equivalent to lowest RMSE), and exports
    salary_model.joblib and model_metrics.json.
    The model is trained on salary in USD; INR / LPA are derived at display time.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    df = load_dataset()
    if df is None:
        return False

    model_df = df.sample(n=min(MODEL_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    X = model_df[FEATURES].copy()
    y = model_df["salary_in_usd"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    def make_pipeline(regressor):
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUM_COLS),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
        ])
        return Pipeline([("preprocessor", preprocessor), ("model", regressor)])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=100.0, max_iter=5000),
    }

    results, fitted = [], {}
    for name, reg in candidates.items():
        pipe = make_pipeline(reg)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results.append({
            "Model": name,
            "Test R²": round(float(r2_score(y_test, pred)), 4),
            "Test MAE ($)": round(float(mae), 2),
            "Test MAE (₹)": round(float(mae * DEFAULT_USD_TO_INR), 2),
            "Test MAE (LPA)": round(float(mae * DEFAULT_USD_TO_INR / 100000.0), 2),
            "Test RMSE ($)": round(rmse, 2),
            "Test RMSE (₹)": round(rmse * DEFAULT_USD_TO_INR, 2),
            "Test RMSE (LPA)": round(rmse * DEFAULT_USD_TO_INR / 100000.0, 2),
        })
        fitted[name] = pipe

    # Best model = highest R² on the held-out test set (closest to 1)
    best = max(results, key=lambda r: r["Test R²"])
    best_pipe = fitted[best["Model"]]

    # Permutation importance on the held-out set (works for any regressor)
    perm = permutation_importance(
        best_pipe, X_test, y_test, scoring="r2", n_repeats=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    raw = np.clip(perm.importances_mean, 0, None)
    total = raw.sum() if raw.sum() > 0 else 1.0
    importances = sorted(
        [{"feature": f, "importance": round(float(v / total), 5)} for f, v in zip(X_test.columns, raw)],
        key=lambda d: d["importance"], reverse=True
    )

    sample = X_test.assign(actual=y_test).sample(n=min(400, len(X_test)), random_state=RANDOM_STATE)
    sample_pred = best_pipe.predict(sample[FEATURES])
    samples = [
        {"actual_lpa": round(float(a * DEFAULT_USD_TO_INR / 100000.0), 3),
         "predicted_lpa": round(float(p * DEFAULT_USD_TO_INR / 100000.0), 3)}
        for a, p in zip(sample["actual"], sample_pred)
    ]

    metrics = {
        "pipeline_version": PIPELINE_VERSION,
        "best_model_name": best["Model"],
        "best_model_metrics": best,
        "model_comparison": results,
        "feature_importances": importances,
        "dataset_stats": {
            "total_records": int(len(df)),
            "median_salary_usd": float(df["salary_in_usd"].median()),
            "median_salary_inr": float(df["salary_in_inr"].median()),
            "median_salary_lpa": float(df["salary_in_lpa"].median()),
            "modeling_rows": int(len(model_df)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "features": {
            "numeric": NUM_COLS,
            "categorical": CAT_COLS,
            "categorical_options": {c: sorted(df[c].unique().tolist()) for c in CAT_COLS},
        },
        "actual_vs_predicted_samples": samples,
    }

    joblib.dump(best_pipe, MODEL_FILE)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return True

df_data = load_dataset()
model_pipeline = load_trained_model()
metrics_data = load_model_metrics()

# -----------------------------------------------------------------------------
# UI Header & Currency Notice
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
        <span class="badge-pill pill-emerald">🇮🇳 Target: Indian Rupees (₹)</span>
        <span class="badge-pill pill-purple">⚡ Benchmark: ₹85 / USD</span>
        <span class="badge-pill pill-amber">🏆 Best Regression Model Selected</span>
    </div>
    <h1 class="hero-title">💼 SalarySense — Job Salary Intelligence</h1>
    <p class="hero-subtitle">
        Explore how experience, education, skills, industry and location shape tech compensation, and predict annual packages converted into Indian Rupees (₹ and LPA) using trained Scikit-learn regressors.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="rate-banner">
    <div style="display: flex; align-items: flex-start; gap: 12px;">
        <span style="font-size: 1.4rem;">💡</span>
        <div>
            <strong style="color: #B45309; font-size: 1rem;">Exchange Rate Benchmark: 1 USD = ₹{DEFAULT_USD_TO_INR:.2f} INR (Illustrative Rate)</strong><br>
            <span style="color: #334155; font-size: 0.9rem;">
                Dataset salaries are treated as USD and converted to give an intuitive reference in the Indian numbering system (₹ and LPA). Note: direct currency conversion does not imply purchasing power parity (PPP) or domestic Indian hiring budgets.
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if df_data is None:
    st.markdown(f"""
    <div class="info-callout">
        <strong style="font-size: 1.05rem;">⚠️ Dataset Not Found</strong><br>
        Place <code>job_salary_prediction_dataset.csv</code> in the same folder as <code>app.py</code>:
        <code>{APP_DIR}</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# Navigation Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    section = st.radio(
        "Go to Section:",
        ["📊 Overview", "🔮 Predict Salary", "🔍 Salary Explorer", "📈 Model Insights"],
        index=0
    )

# -----------------------------------------------------------------------------
# SECTION 1: OVERVIEW
# -----------------------------------------------------------------------------
if section == "📊 Overview":
    st.markdown("### 📌 Executive Summary & Key Indicators")

    total_records = len(df_data)
    median_salary_inr = df_data["salary_in_inr"].median()
    median_salary_lpa = df_data["salary_in_lpa"].median()

    best_r2 = f"{metrics_data['best_model_metrics']['Test R²']:.4f}" if metrics_data else "—"
    best_model_name = metrics_data["best_model_name"] if metrics_data else "Not trained yet"
    best_mae_lpa = f"{metrics_data['best_model_metrics']['Test MAE (LPA)']:.2f}" if metrics_data else "—"
    mae_display = f"₹{best_mae_lpa} LPA" if metrics_data else "—"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card card-cyan">
            <div class="metric-icon-wrap icon-cyan">📁</div>
            <div>
                <div class="metric-title">Dataset Records</div>
                <div class="metric-value text-cyan">{total_records:,}</div>
                <div class="metric-sub">Cleaned Roles Analyzed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card card-emerald">
            <div class="metric-icon-wrap icon-emerald">💰</div>
            <div>
                <div class="metric-title">Median Salary (INR)</div>
                <div class="metric-value text-emerald">{format_lpa(median_salary_inr)}</div>
                <div class="metric-sub">{format_inr(median_salary_inr)} / yr</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card card-purple">
            <div class="metric-icon-wrap icon-purple">⚡</div>
            <div>
                <div class="metric-title">Champion Model R²</div>
                <div class="metric-value text-purple">{best_r2}</div>
                <div class="metric-sub">{best_model_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card card-coral">
            <div class="metric-icon-wrap icon-coral">🎯</div>
            <div>
                <div class="metric-title">Test MAE (Error)</div>
                <div class="metric-value text-coral">{mae_display}</div>
                <div class="metric-sub">Mean Absolute Error</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 📊 Salary Distribution (in LPA at ₹85/USD)")
        fig_dist = px.histogram(
            df_data, x="salary_in_lpa", nbins=45,
            color_discrete_sequence=["#06B6D4"], opacity=0.9,
            labels={"salary_in_lpa": "Annual Salary (₹ Lakhs Per Annum)"}
        )
        fig_dist.add_vline(
            x=median_salary_lpa, line_dash="dash", line_color="#E11D48", line_width=3,
            annotation_text=f"Median: ₹{median_salary_lpa:.1f} LPA",
            annotation_position="top right",
            annotation_font_color="#E11D48", annotation_font_size=12
        )
        st.plotly_chart(style_fig(fig_dist), width="stretch")

    with chart_col2:
        st.markdown("#### 🎓 Salary Range by Education Level")
        edu_colors = {
            "High School": "#F43F5E", "Diploma": "#F59E0B", "Bachelor": "#10B981",
            "Master": "#06B6D4", "PhD": "#8B5CF6"
        }
        fig_box = px.box(
            df_data, x="education_level", y="salary_in_lpa",
            category_orders={"education_level": EDU_ORDER},
            color="education_level", color_discrete_map=edu_colors,
            labels={"education_level": "Education Level", "salary_in_lpa": "Salary (₹ LPA)"}
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_box), width="stretch")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### 📈 Median Salary by Years of Experience")
        by_exp = df_data.groupby("experience_years")["salary_in_lpa"].median().reset_index()
        fig_exp = px.line(
            by_exp, x="experience_years", y="salary_in_lpa", markers=True,
            color_discrete_sequence=["#8B5CF6"],
            labels={"experience_years": "Years of Experience", "salary_in_lpa": "Median Salary (₹ LPA)"}
        )
        fig_exp.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(style_fig(fig_exp, height=360), width="stretch")

    with chart_col4:
        st.markdown("#### 🌍 Median Salary by Location")
        by_loc = (df_data.groupby("location")["salary_in_lpa"].median()
                  .reset_index().sort_values("salary_in_lpa", ascending=False))
        by_loc["label"] = by_loc["salary_in_lpa"].apply(lambda x: f"₹{x:.1f}")
        fig_loc = px.bar(
            by_loc, x="location", y="salary_in_lpa", color="location",
            color_discrete_sequence=px.colors.qualitative.Vivid, text="label",
            labels={"location": "Location", "salary_in_lpa": "Median Salary (₹ LPA)"}
        )
        fig_loc.update_traces(textposition="outside")
        fig_loc.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_loc, height=360), width="stretch")

    st.markdown("#### 🏆 Job Roles Ranked by Median Salary (₹ LPA)")
    top_roles = (df_data.groupby("job_title")["salary_in_lpa"].agg(["median", "count"])
                 .reset_index().sort_values(by="median", ascending=False))
    top_roles["median_formatted"] = top_roles["median"].apply(lambda x: f" ₹{x:.1f} LPA")

    vibrant_palette = [
        "#F43F5E", "#EC4899", "#8B5CF6", "#6366F1", "#3B82F6", "#0EA5E9",
        "#06B6D4", "#10B981", "#84CC16", "#F59E0B", "#F97316", "#14B8A6"
    ]
    fig_bar = px.bar(
        top_roles, x="median", y="job_title", orientation="h",
        color="job_title", color_discrete_sequence=vibrant_palette,
        labels={"median": "Median Salary (₹ LPA)", "job_title": "Job Role"},
        text="median_formatted"
    )
    fig_bar.update_traces(textposition="outside", textfont=dict(size=11, color="#0F172A", weight=700))
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed", gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
        xaxis=AXIS_STYLE, showlegend=False,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=80, t=20, b=20), height=470
    )
    st.plotly_chart(fig_bar, width="stretch")

# -----------------------------------------------------------------------------
# SECTION 2: PREDICT SALARY
# -----------------------------------------------------------------------------
elif section == "🔮 Predict Salary":
    st.markdown("### 🔮 Predict Job Compensation")
    st.write("Configure candidate profile details matching the trained Scikit-learn pipeline features.")

    if model_pipeline is None:
        st.markdown("""
        <div class="info-callout">
            <strong style="font-size: 1.05rem;">⚠️ Model Artifact Missing (<code>salary_model.joblib</code>)</strong><br>
            The trained model was not found in the application folder. Click the button below to train all
            the Linear, Ridge and Lasso regression models and save the best one (takes under a minute).
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Train & Generate salary_model.joblib Locally Now", type="primary"):
            with st.spinner("Training Linear, Ridge and Lasso regression pipelines..."):
                success = train_model_locally()
            if success:
                st.success("✅ Model trained and saved as `salary_model.joblib`! Reloading dashboard...")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Failed to load dataset. Make sure `job_salary_prediction_dataset.csv` is next to app.py.")

    opts = {c: sorted(df_data[c].unique().tolist()) for c in CAT_COLS}
    edu_opts = [e for e in EDU_ORDER if e in opts["education_level"]]
    size_opts = [s for s in SIZE_ORDER if s in opts["company_size"]]
    remote_opts = [r for r in REMOTE_ORDER if r in opts["remote_work"]]

    with st.form("salary_prediction_form"):
        st.markdown('<h5 style="color: #0284C7;">1. Role & Experience</h5>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            job_title = st.selectbox(
                "Job Title / Role", options=opts["job_title"],
                index=opts["job_title"].index("Data Scientist") if "Data Scientist" in opts["job_title"] else 0
            )
        with col_b:
            experience_years = st.slider(
                "Years of Experience", min_value=0, max_value=20, value=5,
                help="Total years of professional experience."
            )
        with col_c:
            education_level = st.selectbox(
                "Education Level", options=edu_opts,
                format_func=lambda x: {
                    "High School": "🏫 High School",
                    "Diploma": "📜 Diploma",
                    "Bachelor": "🎓 Bachelor's Degree",
                    "Master": "📘 Master's Degree",
                    "PhD": "🔬 PhD / Doctorate",
                }.get(x, x),
                index=edu_opts.index("Bachelor") if "Bachelor" in edu_opts else 0
            )

        st.markdown('<h5 style="color: #7C3AED; margin-top: 14px;">2. Skills & Certifications</h5>', unsafe_allow_html=True)
        col_d, col_e = st.columns(2)
        with col_d:
            skills_count = st.slider(
                "Number of Skills", min_value=1, max_value=int(df_data["skills_count"].max()), value=10,
                help="Count of relevant technical skills."
            )
        with col_e:
            certifications = st.slider(
                "Number of Certifications", min_value=0, max_value=int(df_data["certifications"].max()), value=2
            )

        st.markdown('<h5 style="color: #059669; margin-top: 14px;">3. Company, Location & Work Mode</h5>', unsafe_allow_html=True)
        col_f, col_g = st.columns(2)
        with col_f:
            industry = st.selectbox(
                "Industry", options=opts["industry"],
                index=opts["industry"].index("Technology") if "Technology" in opts["industry"] else 0
            )
            company_size = st.selectbox(
                "Company Size", options=size_opts,
                format_func=lambda x: {
                    "Startup": "🚀 Startup",
                    "Small": "🌱 Small",
                    "Medium": "🏢 Medium",
                    "Large": "🏬 Large",
                    "Enterprise": "🏰 Enterprise",
                }.get(x, x),
                index=size_opts.index("Medium") if "Medium" in size_opts else 0
            )
        with col_g:
            location = st.selectbox(
                "Location", options=opts["location"],
                index=opts["location"].index("USA") if "USA" in opts["location"] else 0,
                help="Country of the role (or 'Remote')."
            )
            remote_work = st.select_slider(
                "Remote Work Arrangement", options=remote_opts,
                value="Hybrid" if "Hybrid" in remote_opts else remote_opts[0],
                format_func=lambda x: {
                    "No": "🏢 On-site / Office",
                    "Hybrid": "🌐 Hybrid",
                    "Yes": "🏠 Fully Remote",
                }.get(x, x)
            )

        submit_btn = st.form_submit_button("⚡ Calculate Predicted Salary (INR & LPA)", width="stretch")

    if submit_btn:
        input_data = pd.DataFrame([{
            "experience_years": int(experience_years),
            "skills_count": int(skills_count),
            "certifications": int(certifications),
            "job_title": str(job_title),
            "education_level": str(education_level),
            "industry": str(industry),
            "company_size": str(company_size),
            "location": str(location),
            "remote_work": str(remote_work),
        }])[FEATURES]

        if model_pipeline is not None:
            try:
                predicted_usd = max(0.0, float(model_pipeline.predict(input_data)[0]))
                predicted_inr = predicted_usd * DEFAULT_USD_TO_INR
                predicted_lpa = predicted_inr / 100000.0

                st.markdown(f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;">
                            ✨ Estimated Annual Package
                        </span>
                        <span class="badge-pill pill-emerald" style="font-size: 0.82rem;">⚡ Real-time ML Inference</span>
                    </div>
                    <div class="prediction-value-gradient">{format_lpa(predicted_inr)}</div>
                    <div style="display: flex; gap: 10px; margin-top: 14px; margin-bottom: 20px; flex-wrap: wrap;">
                        <span class="badge-pill pill-emerald">🇮🇳 Annual: {format_inr(predicted_inr)}</span>
                        <span class="badge-pill pill-cyan">📅 Monthly: {format_monthly_inr(predicted_inr)}</span>
                        <span class="badge-pill pill-amber">💵 USD Eqv: ${predicted_usd:,.0f}</span>
                        <span class="badge-pill pill-purple">🎯 Role: {job_title}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 14px;">
                        💡 <strong>Illustrative Conversion (₹85/USD):</strong> Represents a global benchmark translated to INR. Domestic Indian tech hiring budgets follow regional market pay scales.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 📊 Benchmark Context Comparison")
                median_ref = df_data["salary_in_lpa"].median()
                diff_pct = ((predicted_lpa - median_ref) / median_ref) * 100
                diff_sign = "+" if diff_pct >= 0 else ""
                diff_color = "#059669" if diff_pct >= 0 else "#E11D48"
                st.markdown(
                    f"This prediction is <strong style='color: {diff_color};'>{diff_sign}{diff_pct:.1f}%</strong> "
                    f"relative to the overall dataset median of <strong style='color: #0284C7;'>₹{median_ref:.2f} LPA</strong>.",
                    unsafe_allow_html=True
                )

                same_role = df_data[df_data["job_title"] == job_title]["salary_in_lpa"]
                if len(same_role) > 0:
                    role_median = same_role.median()
                    role_diff = ((predicted_lpa - role_median) / role_median) * 100
                    rd_sign = "+" if role_diff >= 0 else ""
                    rd_color = "#059669" if role_diff >= 0 else "#E11D48"
                    st.markdown(
                        f"Compared with other <strong>{job_title}</strong> records, it is "
                        f"<strong style='color: {rd_color};'>{rd_sign}{role_diff:.1f}%</strong> "
                        f"versus that role's median of <strong style='color: #0284C7;'>₹{role_median:.2f} LPA</strong>.",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error executing prediction pipeline: {str(e)}")
        else:
            st.error("Cannot predict: `salary_model.joblib` was not found. Please train the model using the button above.")

# -----------------------------------------------------------------------------
# SECTION 3: SALARY EXPLORER
# -----------------------------------------------------------------------------
elif section == "🔍 Salary Explorer":
    st.markdown("### 🔍 Interactive Salary Explorer")
    st.write("Filter, compare, and analyze salary records converted into Indian Rupees (₹ and LPA).")

    st.markdown("##### 🎯 Filter Records")
    f_col1, f_col2, f_col3 = st.columns(3)
    f_col4, f_col5, f_col6 = st.columns(3)

    with f_col1:
        sel_roles = st.multiselect("Job Title(s)", options=sorted(df_data["job_title"].unique()), default=[])
    with f_col2:
        sel_edu = st.multiselect("Education Level",
                                 options=[e for e in EDU_ORDER if e in df_data["education_level"].unique()], default=[])
    with f_col3:
        sel_industry = st.multiselect("Industry", options=sorted(df_data["industry"].unique()), default=[])
    with f_col4:
        sel_sizes = st.multiselect("Company Size",
                                   options=[s for s in SIZE_ORDER if s in df_data["company_size"].unique()], default=[])
    with f_col5:
        sel_loc = st.multiselect("Location", options=sorted(df_data["location"].unique()), default=[])
    with f_col6:
        sel_remote = st.multiselect("Remote Work",
                                    options=[r for r in REMOTE_ORDER if r in df_data["remote_work"].unique()],
                                    format_func=lambda x: {"No": "On-site", "Hybrid": "Hybrid", "Yes": "Fully Remote"}.get(x, x),
                                    default=[])

    exp_min, exp_max = int(df_data["experience_years"].min()), int(df_data["experience_years"].max())
    sel_exp = st.slider("Years of Experience", min_value=exp_min, max_value=exp_max, value=(exp_min, exp_max))

    filtered_df = df_data.copy()
    if sel_roles:
        filtered_df = filtered_df[filtered_df["job_title"].isin(sel_roles)]
    if sel_edu:
        filtered_df = filtered_df[filtered_df["education_level"].isin(sel_edu)]
    if sel_industry:
        filtered_df = filtered_df[filtered_df["industry"].isin(sel_industry)]
    if sel_sizes:
        filtered_df = filtered_df[filtered_df["company_size"].isin(sel_sizes)]
    if sel_loc:
        filtered_df = filtered_df[filtered_df["location"].isin(sel_loc)]
    if sel_remote:
        filtered_df = filtered_df[filtered_df["remote_work"].isin(sel_remote)]
    filtered_df = filtered_df[filtered_df["experience_years"].between(sel_exp[0], sel_exp[1])]

    st.markdown(
        f"**Found <span style='color: #0284C7;'>{len(filtered_df):,}</span> matching records out of {len(df_data):,} total.**",
        unsafe_allow_html=True
    )

    if len(filtered_df) > 0:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card card-cyan"><div>
                <div class="metric-title">Filtered Count</div>
                <div class="metric-value text-cyan">{len(filtered_df):,}</div>
            </div></div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card card-emerald"><div>
                <div class="metric-title">Median Salary</div>
                <div class="metric-value text-emerald">{format_lpa(filtered_df["salary_in_inr"].median())}</div>
            </div></div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card card-purple"><div>
                <div class="metric-title">Average Salary</div>
                <div class="metric-value text-purple">{format_lpa(filtered_df["salary_in_inr"].mean())}</div>
            </div></div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card card-coral"><div>
                <div class="metric-title">Maximum Salary</div>
                <div class="metric-value text-coral">{format_lpa(filtered_df["salary_in_inr"].max())}</div>
            </div></div>
            """, unsafe_allow_html=True)

        # Large selections are sampled for plotting speed (stats above use all rows)
        plot_df = filtered_df.sample(n=min(len(filtered_df), 20000), random_state=1)

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.markdown("###### Salary Distribution in Selection (LPA)")
            fig_f_dist = px.histogram(
                filtered_df, x="salary_in_lpa", nbins=40,
                color_discrete_sequence=["#8B5CF6"], labels={"salary_in_lpa": "Salary (₹ LPA)"}
            )
            st.plotly_chart(style_fig(fig_f_dist, height=310, margin=dict(l=20, r=20, t=20, b=20)),
                            width="stretch")

        with exp_c2:
            st.markdown("###### Compensation by Remote Arrangement")
            remote_color_map = {"No": "#3B82F6", "Hybrid": "#8B5CF6", "Yes": "#10B981"}
            fig_f_remote = px.box(
                plot_df, x="remote_work", y="salary_in_lpa", color="remote_work",
                category_orders={"remote_work": REMOTE_ORDER},
                labels={"remote_work": "Remote Work", "salary_in_lpa": "Salary (₹ LPA)"},
                color_discrete_map=remote_color_map
            )
            fig_f_remote.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig_f_remote, height=310, margin=dict(l=20, r=20, t=20, b=20)),
                            width="stretch")

        exp_c3, exp_c4 = st.columns(2)
        with exp_c3:
            st.markdown("###### Salary vs. Experience (Median by Year)")
            sel_by_exp = filtered_df.groupby("experience_years")["salary_in_lpa"].median().reset_index()
            fig_f_exp = px.line(
                sel_by_exp, x="experience_years", y="salary_in_lpa", markers=True,
                color_discrete_sequence=["#06B6D4"],
                labels={"experience_years": "Years of Experience", "salary_in_lpa": "Median Salary (₹ LPA)"}
            )
            fig_f_exp.update_traces(line=dict(width=3))
            st.plotly_chart(style_fig(fig_f_exp, height=310, margin=dict(l=20, r=20, t=20, b=20)),
                            width="stretch")

        with exp_c4:
            st.markdown("###### Median Salary by Industry")
            sel_by_ind = (filtered_df.groupby("industry")["salary_in_lpa"].median()
                          .reset_index().sort_values("salary_in_lpa", ascending=False))
            fig_f_ind = px.bar(
                sel_by_ind, x="industry", y="salary_in_lpa", color="industry",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"industry": "Industry", "salary_in_lpa": "Median Salary (₹ LPA)"}
            )
            fig_f_ind.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig_f_ind, height=310, margin=dict(l=20, r=20, t=20, b=20)),
                            width="stretch")

        st.markdown("##### 📋 Filtered Records Data Table")
        table_src = filtered_df.sort_values("salary_in_usd", ascending=False).head(1000)
        display_df = table_src[[
            "job_title", "experience_years", "education_level", "skills_count", "certifications",
            "industry", "company_size", "location", "remote_work", "salary_in_usd", "salary_in_inr"
        ]].copy()
        display_df["salary_in_inr_fmt"] = display_df["salary_in_inr"].apply(format_inr)
        display_df["salary_in_lpa_fmt"] = display_df["salary_in_inr"].apply(format_lpa)
        display_df["salary_in_usd_fmt"] = display_df["salary_in_usd"].apply(lambda x: f"${x:,.0f}")

        st.caption(f"Showing the top {len(display_df):,} highest-paid records from the current selection.")
        st.dataframe(
            display_df[[
                "job_title", "experience_years", "education_level", "skills_count", "certifications",
                "industry", "company_size", "location", "remote_work",
                "salary_in_usd_fmt", "salary_in_inr_fmt", "salary_in_lpa_fmt"
            ]].rename(columns={
                "job_title": "Job Title",
                "experience_years": "Exp (yrs)",
                "education_level": "Education",
                "skills_count": "Skills",
                "certifications": "Certs",
                "industry": "Industry",
                "company_size": "Size",
                "location": "Location",
                "remote_work": "Remote",
                "salary_in_usd_fmt": "Salary (USD)",
                "salary_in_inr_fmt": "Salary (INR ₹)",
                "salary_in_lpa_fmt": "Salary (LPA)"
            }),
            width="stretch",
            height=350
        )
    else:
        st.warning("No records matched the selected combination of filters. Please broaden your selection.")

# -----------------------------------------------------------------------------
# SECTION 4: MODEL INSIGHTS
# -----------------------------------------------------------------------------
elif section == "📈 Model Insights":
    st.markdown("### 📈 Model Evaluation, Comparisons & Caveats")
    st.write("Examine regression algorithm performance, actual vs predicted evaluations, and key methodology details.")

    if metrics_data and "model_comparison" in metrics_data:
        comp_df = pd.DataFrame(metrics_data["model_comparison"])

        st.markdown("##### 🏆 Algorithm Performance Comparison (Held-Out Test Set)")
        st.dataframe(
            comp_df.rename(columns={
                "Model": "Algorithm",
                "Test R²": "R² Score",
                "Test MAE ($)": "MAE ($)",
                "Test MAE (₹)": "MAE (₹)",
                "Test MAE (LPA)": "MAE (₹ LPA)",
                "Test RMSE ($)": "RMSE ($)",
                "Test RMSE (₹)": "RMSE (₹)",
                "Test RMSE (LPA)": "RMSE (₹ LPA)"
            }),
            width="stretch"
        )

        algo_colors = {
            "Linear Regression": "#0EA5E9",
            "Ridge Regression": "#10B981",
            "Lasso Regression": "#F59E0B",
            "Decision Tree": "#F43F5E",
            "Random Forest": "#8B5CF6",
            "Gradient Boosting": "#06B6D4",
        }

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### Test R² Score Comparison")
            fig_r2 = px.bar(comp_df, x="Model", y="Test R²", color="Model",
                            color_discrete_map=algo_colors, text_auto=".3f")
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig_r2, height=310, margin=dict(l=20, r=20, t=20, b=20)),
                            width="stretch")

        with c2:
            st.markdown("###### Test Mean Absolute Error (MAE in LPA)")
            fig_mae = px.bar(comp_df, x="Model", y="Test MAE (LPA)", color="Model",
                             color_discrete_map=algo_colors, text_auto=".2f")
            fig_mae.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig_mae, height=310, margin=dict(l=20, r=20, t=20, b=20)),
                            width="stretch")
    else:
        st.info("Model metrics (`model_metrics.json`) are not present yet. Go to **🔮 Predict Salary** and click the training button to generate them.")

    if metrics_data and metrics_data.get("feature_importances"):
        st.markdown("##### 🧬 What Drives Salary? (Feature Importance)")
        fi_df = pd.DataFrame(metrics_data["feature_importances"])
        pretty = {
            "experience_years": "Years of Experience", "skills_count": "Skills Count",
            "certifications": "Certifications", "job_title": "Job Title",
            "education_level": "Education Level", "industry": "Industry",
            "company_size": "Company Size", "location": "Location", "remote_work": "Remote Work"
        }
        fi_df["Feature"] = fi_df["feature"].map(pretty).fillna(fi_df["feature"])
        fi_df["pct"] = fi_df["importance"] * 100
        fig_fi = px.bar(
            fi_df, x="pct", y="Feature", orientation="h", color="Feature",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"pct": "Importance (%)"}, text=fi_df["pct"].apply(lambda v: f"{v:.1f}%")
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed", gridcolor="#E2E8F0", tickfont=dict(color="#334155")),
            xaxis=AXIS_STYLE, showlegend=False,
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            margin=dict(l=20, r=60, t=20, b=20), height=360
        )
        st.plotly_chart(fig_fi, width="stretch")

    if metrics_data and "actual_vs_predicted_samples" in metrics_data:
        st.markdown("##### 🎯 Actual vs. Predicted Salaries on Held-Out Test Set")
        samples_df = pd.DataFrame(metrics_data["actual_vs_predicted_samples"])

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=samples_df["actual_lpa"], y=samples_df["predicted_lpa"], mode="markers",
            marker=dict(
                size=8, color=samples_df["actual_lpa"], colorscale="Viridis", showscale=True,
                colorbar=dict(title="LPA", tickfont=dict(color="#334155")),
                line=dict(width=1, color="white")
            ),
            name="Test Predictions"
        ))
        min_v = min(samples_df["actual_lpa"].min(), samples_df["predicted_lpa"].min())
        max_v = max(samples_df["actual_lpa"].max(), samples_df["predicted_lpa"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode="lines",
            line=dict(color="#E11D48", dash="dash", width=2.5), name="Ideal Fit (y = x)"
        ))
        fig_scatter.update_layout(
            xaxis_title="Actual Salary (₹ LPA)", yaxis_title="Predicted Salary (₹ LPA)"
        )
        st.plotly_chart(style_fig(fig_scatter, height=390, margin=dict(l=20, r=20, t=20, b=20)),
                        width="stretch")

    st.markdown("##### 🔬 Preprocessing, Leakage Prevention & Key Limitations")
    with st.expander("1. Leakage-Free Preprocessing Architecture", expanded=True):
        st.markdown("""
        - **Pipeline Encapsulation:** Numeric features (`experience_years`, `skills_count`, `certifications`) are median-imputed and scaled via `StandardScaler`; categorical features (`job_title`, `education_level`, `industry`, `company_size`, `location`, `remote_work`) are most-frequent-imputed and encoded via `OneHotEncoder(handle_unknown='ignore')`. Everything lives inside one pipeline, so new inputs get identical treatment.
        - **Train-Test Isolation:** Exact duplicates and invalid values are removed, then a reproducible 50,000-row sample (`random_state=42`) is split 80/20. Imputers, scalers and encoders are fitted only on the training portion and evaluated on the held-out test portion, preventing data leakage.
        - **Model Comparison:** The champion is the model with the highest held-out R² (equivalently, the lowest RMSE) among the candidate regressors shown in the comparison table above.
        - **Target Handling:** Models are trained on salary in USD; INR and LPA values are derived at display time using the fixed ₹85/USD benchmark.
        """)

    with st.expander("2. Robust Handling of Unseen Categorical Values"):
        st.markdown("""
        - Real-world inputs may include rare or novel job titles, industries or locations.
        - By configuring `OneHotEncoder(handle_unknown='ignore')`, any unseen category is safely ignored without crashing or corrupting predictions.
        """)

    with st.expander("3. Crucial Limitations of Currency-Converted Salaries in the Indian Context"):
        st.markdown("""
        - **Dataset Nature:** This dataset appears to be uniformly structured (similar spread across roles, industries and locations), so patterns may be cleaner than in real-world hiring data and should be treated as indicative rather than definitive.
        - **Observational Data:** The data cannot prove causality, important salary drivers may be missing, and `location` may encode market or currency differences. With very large samples, even tiny effects become statistically significant.
        - **Nominal FX vs. Purchasing Power Parity (PPP):** Direct conversion using ₹85/USD provides an illustrative financial equivalent, but **does not imply that an identical role in India commands that dollar-converted salary**. Domestic Indian tech salaries reflect local market dynamics, cost of living, and domestic hiring budgets.
        - **Currency Assumption:** The dataset does not state its currency; salaries are assumed to be in USD.
        """)