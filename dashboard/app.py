"""
Streamlit Dashboard for UPI Fraud Detection System

Displays:
- Real-time risk scores
- SHAP waterfall explanations for alerts
- Drift monitoring alerts
- Model performance metrics
- Retraining strategy comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from data_prep import load_ieee_cis, load_paysim, prepare_features_target
from models import evaluate_model, get_model
from drift_detection import DriftMonitor, population_stability_index, alert_key
from shap_explainer import SHAPExplainer
from pipeline import stage_train, stage_drift, stage_retrain

# API client for integration testing
from starlette.testclient import TestClient
from api import app as api_app
api_client = TestClient(api_app)

# Page config
st.set_page_config(
    page_title="UPI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths (resolved relative to this file so it works from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
ALERT_STATUS_PATH = REPORTS_DIR / 'alert_status.json'


# ── Alert status persistence ──
def load_alert_status():
    """Load alert acknowledgment/resolution status from disk."""
    if ALERT_STATUS_PATH.exists():
        try:
            with open(ALERT_STATUS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_alert_status(status_dict):
    """Persist alert status to disk."""
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ALERT_STATUS_PATH, 'w') as f:
            json.dump(status_dict, f, indent=2)
    except Exception as exc:
        st.error(f"Failed to save alert status: {exc}")



@st.cache_data
def load_reports():
    """Load all report files."""
    reports = {}
    report_files = {
        'ieee_test_metrics': REPORTS_DIR / 'ieee_test_metrics.csv',
        'paysim_test_metrics': REPORTS_DIR / 'paysim_test_metrics.csv',
        'drift_alerts': REPORTS_DIR / 'drift_alerts_log.csv',
        'shap_global': REPORTS_DIR / 'shap_global_importance.csv',
        'shap_stability': REPORTS_DIR / 'shap_stability.csv',
        'retrain_comparison': REPORTS_DIR / 'retraining_strategy_comparison.csv',
        'delay_impact': REPORTS_DIR / 'delay_impact_summary.csv',
    }
    for name, path in report_files.items():
        if path.exists():
            # CSVs from notebooks use index_col=0 for model names
            if name in ('ieee_test_metrics', 'paysim_test_metrics'):
                df = pd.read_csv(path, index_col=0)
                # Convert index to 'model' column for downstream use
                df = df.reset_index().rename(columns={'index': 'model'})
                reports[name] = df
            else:
                reports[name] = pd.read_csv(path)
        else:
            reports[name] = pd.DataFrame()
    return reports


# Session state initialization
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "alert_status" not in st.session_state:
    st.session_state.alert_status = load_alert_status()

# Custom CSS - Modern AI/3D Style Dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        /* Light theme tokens */
        --bg-primary: #fafafa;
        --bg-secondary: #ffffff;
        --bg-tertiary: #f5f5f5;
        --bg-card: rgba(255, 255, 255, 0.85);
        --bg-card-hover: rgba(255, 255, 255, 0.95);
        --border-color: rgba(148, 163, 184, 0.2);
        --border-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #06b6d4;
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        --shadow-sm: 0 1px 2px 0 rgba(15, 23, 42, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -2px rgba(15, 23, 42, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(15, 23, 42, 0.1), 0 4px 6px -4px rgba(15, 23, 42, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        --shadow-3d: 0 25px 50px -12px rgba(99, 102, 241, 0.25), 0 0 0 1px rgba(99, 102, 241, 0.1);
        --shadow-glow: 0 0 40px -10px rgba(99, 102, 241, 0.4);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 400ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-theme="dark"], .dark-mode {
        --bg-primary: #030712;
        --bg-secondary: #0f172a;
        --bg-tertiary: #1e293b;
        --bg-card: rgba(15, 23, 42, 0.85);
        --bg-card-hover: rgba(30, 41, 59, 0.95);
        --border-color: rgba(148, 163, 184, 0.15);
        --border-glow: rgba(129, 140, 248, 0.4);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --accent-primary: #818cf8;
        --accent-secondary: #a78bfa;
        --accent-tertiary: #22d3ee;
        --accent-gradient: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #22d3ee 100%);
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
        --shadow-3d: 0 25px 50px -12px rgba(129, 140, 248, 0.3), 0 0 0 1px rgba(129, 140, 248, 0.15);
        --shadow-glow: 0 0 40px -10px rgba(129, 140, 248, 0.5);
    }

    /* Global styles */
    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Animated gradient background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse 50% 30% at 50% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 60%);
        pointer-events: none;
        z-index: -1;
        animation: gradientShift 20s ease-in-out infinite;
    }

    @keyframes gradientShift {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 1; }
        25% { transform: translate(-2%, 1%) scale(1.02); opacity: 0.9; }
        50% { transform: translate(1%, -1%) scale(0.98); opacity: 1; }
        75% { transform: translate(-1%, -2%) scale(1.01); opacity: 0.95; }
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    h1 { font-size: clamp(2rem, 5vw, 3rem) !important; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    h2 { font-size: clamp(1.5rem, 3vw, 2.25rem) !important; }
    h3 { font-size: clamp(1.25rem, 2.5vw, 1.75rem) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        backdrop-filter: blur(20px) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--accent-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    /* Cards - Glassmorphism with 3D depth */
    .glass-card {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
        box-shadow: var(--shadow-md) !important;
        transition: all var(--transition-normal) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        opacity: 0;
        transition: opacity var(--transition-normal);
    }

    .glass-card:hover {
        transform: translateY(-4px) scale(1.01) !important;
        box-shadow: var(--shadow-3d) !important;
        border-color: var(--border-glow) !important;
        background: var(--bg-card-hover) !important;
    }

    .glass-card:hover::before {
        opacity: 1;
    }

    .glass-card-elevated {
        box-shadow: var(--shadow-xl) !important;
        transform: translateZ(20px);
    }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
        box-shadow: var(--shadow-md) !important;
        transition: all var(--transition-normal) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 3px;
        background: var(--accent-gradient);
        transform: translateX(-50%);
        transition: width var(--transition-normal);
        border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    }

    .metric-card:hover {
        transform: translateY(-6px) rotateX(2deg) !important;
        box-shadow: var(--shadow-3d) !important;
        border-color: var(--border-glow) !important;
    }

    .metric-card:hover::after {
        width: 100%;
    }

    .stMetric {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
    }

    .stMetric > div {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.25rem !important;
        transition: all var(--transition-fast) !important;
    }

    .stMetric > div:hover {
        transform: translateY(-3px) !important;
        box-shadow: var(--shadow-lg) !important;
        border-color: var(--border-glow) !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: var(--accent-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        line-height: 1.1 !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--radius-md) !important;
        border: none !important;
        background: var(--accent-gradient) !important;
        color: white !important;
        box-shadow: var(--shadow-md) !important;
        transition: all var(--transition-fast) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left var(--transition-slow);
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: var(--shadow-glow), var(--shadow-lg) !important;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
    }

    /* Secondary button variant */
    .stButton > button[kind="secondary"] {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-card) !important;
        border-color: var(--border-glow) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: var(--bg-tertiary) !important;
        padding: 6px !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--border-color) !important;
        backdrop-filter: blur(20px) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9375rem !important;
        height: 44px !important;
        padding: 0 1.5rem !important;
        border-radius: var(--radius-md) !important;
        background: transparent !important;
        color: var(--text-secondary) !important;
        transition: all var(--transition-fast) !important;
        border: none !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* DataFrames */
    .stDataFrame {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-md) !important;
    }

    .stDataFrame:hover {
        box-shadow: var(--shadow-lg) !important;
        border-color: var(--border-glow) !important;
    }

    /* Alerts */
    .alert-high, .alert-medium, .alert-low {
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.25rem !important;
        margin: 0.5rem 0 !important;
        border: none !important;
        border-left: 4px solid !important;
        transition: all var(--transition-normal) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .alert-high { background: rgba(239, 68, 68, 0.1) !important; border-left-color: #ef4444 !important; }
    .alert-medium { background: rgba(249, 115, 22, 0.1) !important; border-left-color: #f97316 !important; }
    .alert-low { background: rgba(34, 197, 94, 0.1) !important; border-left-color: #22c55e !important; }

    .alert-high:hover, .alert-medium:hover, .alert-low:hover {
        transform: translateX(4px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: all var(--transition-fast) !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        outline: none !important;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background: var(--accent-gradient) !important;
    }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 4px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.25rem !important;
        transition: all var(--transition-fast) !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--border-glow) !important;
        background: var(--bg-card-hover) !important;
    }

    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        padding: 1rem 1.25rem !important;
    }

    /* Radio */
    .stRadio > div {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 0.75rem !important;
        gap: 0.5rem !important;
    }

    .stRadio label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        padding: 0.5rem 1rem !important;
        border-radius: var(--radius-md) !important;
        transition: all var(--transition-fast) !important;
    }

    .stRadio label:hover {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }

    .stRadio input:checked + div {
        background: var(--accent-gradient) !important;
        color: white !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
    }

    /* File uploader */
    .stFileUploader > div > div {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        transition: all var(--transition-fast) !important;
    }

    .stFileUploader > div > div:hover {
        border-color: var(--accent-primary) !important;
        background: var(--bg-card-hover) !important;
    }

    /* Plotly charts */
    .js-plotly-plot {
        background: transparent !important;
        border-radius: var(--radius-lg) !important;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px -5px rgba(99, 102, 241, 0.3); }
        50% { box-shadow: 0 0 40px -5px rgba(99, 102, 241, 0.5); }
    }

    .animate-fade-in-up { animation: fadeInUp 0.6s var(--transition-normal) forwards; }
    .animate-fade-in-scale { animation: fadeInScale 0.4s var(--transition-normal) forwards; }
    .animate-pulse-glow { animation: pulseGlow 2s ease-in-out infinite; }

    /* Staggered animation delays */
    .stagger-1 { animation-delay: 100ms; }
    .stagger-2 { animation-delay: 200ms; }
    .stagger-3 { animation-delay: 300ms; }
    .stagger-4 { animation-delay: 400ms; }
    .stagger-5 { animation-delay: 500ms; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-tertiary); }
    ::-webkit-scrollbar-thumb { background: var(--accent-gradient); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-secondary); }
    ::-webkit-scrollbar-corner { background: var(--bg-tertiary); }

    /* Selection */
    ::selection { background: rgba(99, 102, 241, 0.3); color: var(--text-primary); }

    /* Focus visible */
    :focus-visible {
        outline: 2px solid var(--accent-primary) !important;
        outline-offset: 2px !important;
    }

    /* Reduce motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* Dark mode specific adjustments */
    [data-theme="dark"] .stApp,
    .dark-mode .stApp {
        background: var(--bg-primary) !important;
    }

    [data-theme="dark"] .stApp::before,
    .dark-mode .stApp::before {
        background:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(129, 140, 248, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(167, 139, 250, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 50% 30% at 50% 50%, rgba(34, 211, 238, 0.06) 0%, transparent 60%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load model and data for explanations."""
    model, meta = None, {}
    try:
        model_path = MODELS_DIR / 'ieee_lightgbm.joblib'
        if model_path.exists():
            model, _ = joblib.load(model_path), {}
            meta_path = model_path.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
        else:
            model = None
            meta = {}
    except Exception as e:
        st.error(f"Model load error: {e}")
        model, meta = None, {}

    return model, meta

def apply_fe(df, name):
    """Apply feature engineering."""
    df = df.copy()
    if name == 'ieee-cis':
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
        df['amount_log'] = np.log1p(df['TransactionAmt'])
        df['amount_round'] = (df['TransactionAmt'] == df['TransactionAmt'].round()).astype(int)
        if 'card1' in df.columns:
            df['card1_freq'] = df.groupby('card1')['card1'].transform('count')
    return df

def prepare_xy(df):
    """Prepare X, y from dataframe."""
    drop_cols = ['TransactionID', 'TransactionDT', 'isFraud']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['isFraud'].astype('int8') if 'isFraud' in df.columns else pd.Series(0, index=df.index)
    # encode ALL non-numeric columns (parquet reloads M1-M9 etc. as object)
    for col in X.select_dtypes(exclude=['number']).columns:
        X[col] = X[col].astype('category').cat.codes
    return X.fillna(-999), y

# Sidebar
st.sidebar.title("🛡️ UPI Fraud Detection")
st.sidebar.markdown("---")

# Theme toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# Apply theme via data attribute on document
if dark_mode:
    st.markdown('<script>document.documentElement.setAttribute("data-theme", "dark"); document.body.classList.add("dark-mode")</script>', unsafe_allow_html=True)
else:
    st.markdown('<script>document.documentElement.removeAttribute("data-theme"); document.body.classList.remove("dark-mode")</script>', unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🎯 Risk Scoring", "🤖 Model Comparison", "🚀 Pipeline Control",
     "🔍 SHAP Explanations", "📈 Drift Monitoring",
     "⚙️ Retraining Strategies", "⏱️ Delayed Labels", "📜 Prediction History", "📋 Reports", "🔌 API Integration"]
)

# Load reports
reports = load_reports()
model, model_meta = load_model_and_data()

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "📊 Overview":
    # Hero section with animated gradient text
    st.markdown("""
    <div class="glass-card animate-fade-in-up" style="margin-bottom: 2rem;">
        <div style="text-align: center; padding: 1rem;">
            <h1 style="margin-bottom: 0.5rem;">UPI Fraud Detection</h1>
            <p style="font-size: 1.25rem; color: var(--text-secondary); font-weight: 400; margin: 0;">
                Drift-Aware Explainable ML Framework for Digital Payments
            </p>
            <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 600;">IEEE-CIS</span>
                <span style="color: var(--text-muted);">|</span>
                <span style="background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 600;">PaySim</span>
                <span style="color: var(--text-muted);">|</span>
                <span style="background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 600;">LightGBM</span>
                <span style="color: var(--text-muted);">|</span>
                <span style="background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 600;">SHAP</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards using glass cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="glass-card metric-card animate-fade-in-up stagger-1">
            <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Project</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.75rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.2;">UPI Fraud Detection</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">Drift-Aware Explainable ML</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if not reports['ieee_test_metrics'].empty:
            best_pr_auc = reports['ieee_test_metrics']['pr_auc'].max()
            delta_color = "#22c55e" if best_pr_auc > 0.85 else "#ef4444"
            delta_text = "Target: >0.85" if best_pr_auc > 0.85 else "Below target"
            st.markdown(f"""
            <div class="glass-card metric-card animate-fade-in-up stagger-2">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Best PR-AUC (IEEE-CIS)</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.1;">{best_pr_auc:.4f}</div>
                <div style="font-size: 0.8rem; color: {delta_color}; margin-top: 0.5rem; font-weight: 500;">{delta_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card metric-card animate-fade-in-up stagger-2">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Best PR-AUC</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; color: var(--text-muted); line-height: 1.1;">N/A</div>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if not reports['drift_alerts'].empty:
            total_alerts = len(reports['drift_alerts'])
            st.markdown(f"""
            <div class="glass-card metric-card animate-fade-in-up stagger-3">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Drift Alerts</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.1;">{total_alerts}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">Total alerts logged</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card metric-card animate-fade-in-up stagger-3">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Drift Alerts</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; color: var(--text-muted); line-height: 1.1;">N/A</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if not reports['retrain_comparison'].empty:
            strategies = reports['retrain_comparison']['strategy'].nunique()
            st.markdown(f"""
            <div class="glass-card metric-card animate-fade-in-up stagger-4">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Strategies Compared</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.1;">{strategies}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">Retraining approaches</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card metric-card animate-fade-in-up stagger-4">
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Strategies Compared</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; color: var(--text-muted); line-height: 1.1;">N/A</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project description in glass card
    st.markdown("""
    <div class="glass-card animate-fade-in-up" style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 1.5rem; font-size: 1.5rem;">Project Description</h2>
        <p style="font-size: 1.05rem; line-height: 1.7; color: var(--text-secondary); margin-bottom: 1.5rem;">
            <strong>A Drift-Aware and Explainable Machine Learning Framework for Fraud Detection in UPI Digital Payments under Delayed Labels</strong>
        </p>
        <p style="font-size: 0.95rem; line-height: 1.7; color: var(--text-secondary); margin-bottom: 1rem;">
            This system implements a comprehensive fraud detection pipeline for UPI (Unified Payments Interface) digital payments:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.75rem; margin-top: 1rem;">
            <div style="padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-color); transition: all var(--transition-fast);" onmouseover="this.style.borderColor='var(--border-glow)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border-color)'; this.style.transform='translateY(0)'">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">📊 Data</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">IEEE-CIS & PaySim as UPI proxies</div>
            </div>
            <div style="padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-color); transition: all var(--transition-fast);" onmouseover="this.style.borderColor='var(--border-glow)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border-color)'; this.style.transform='translateY(0)'">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">🤖 Models</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">LR, RF, XGBoost, LightGBM with rolling CV</div>
            </div>
            <div style="padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-color); transition: all var(--transition-fast);" onmouseover="this.style.borderColor='var(--border-glow)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border-color)'; this.style.transform='translateY(0)'">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">📈 Drift Monitoring</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">PSI & KS tests on features & scores</div>
            </div>
            <div style="padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-color); transition: all var(--transition-fast);" onmouseover="this.style.borderColor='var(--border-glow)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border-color)'; this.style.transform='translateY(0)'">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">🔍 Explainability</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">TreeSHAP instance + global importance</div>
            </div>
            <div style="padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-color); transition: all var(--transition-fast);" onmouseover="this.style.borderColor='var(--border-glow)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border-color)'; this.style.transform='translateY(0)'">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">⚙️ Retraining</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Static vs Scheduled vs Drift-Triggered</div>
            </div>
            <div style="padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-color); transition: all var(--transition-fast);" onmouseover="this.style.borderColor='var(--border-glow)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border-color)'; this.style.transform='translateY(0)'">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">⏱️ Delayed Labels</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">1/7/30-day delay simulation</div>
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border-color); display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; color: var(--text-muted); font-size: 0.9rem;">
            <span><strong>Student:</strong> Pavan C (USN: 4VV25MC060)</span>
            <span><strong>College:</strong> Vidyavardhaka College of Engineering</span>
            <span><strong>Course:</strong> MCA, Semester 3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Target achievement with glass cards
    st.markdown("""
    <div class="glass-card animate-fade-in-up" style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 1.5rem; font-size: 1.5rem;">🎯 Target Achievement</h2>
    """, unsafe_allow_html=True)

    if not reports['ieee_test_metrics'].empty:
        best_row = reports['ieee_test_metrics'].loc[reports['ieee_test_metrics']['pr_auc'].idxmax()]

        targets = {
            'PR-AUC > 0.85': (best_row.get('pr_auc', 0) > 0.85, best_row.get('pr_auc', 0)),
            'Precision@100 > 0.70': (best_row.get('precision_at_100', 0) > 0.70, best_row.get('precision_at_100', 0)),
            'FP/100k < 50': (best_row.get('fp_per_100k_at_budget_100', 100) < 50, best_row.get('fp_per_100k_at_budget_100', 100)),
        }

        cols = st.columns(len(targets))
        for i, (target, (passed, value)) in enumerate(targets.items()):
            with cols[i]:
                icon = "✅" if passed else "❌"
                bg_color = "rgba(34, 197, 94, 0.15)" if passed else "rgba(239, 68, 68, 0.15)"
                border_color = "#22c55e" if passed else "#ef4444"
                text_color = "#22c55e" if passed else "#ef4444"
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: var(--radius-lg);
                    padding: 1.5rem;
                    text-align: center;
                    transition: all var(--transition-normal);
                    height: 100%;
                " onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='var(--shadow-lg)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.75rem;">{target}</div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.75rem; font-weight: 700; color: {text_color};">{value:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline diagram
    st.subheader("🔄 Pipeline Architecture")
    st.graphviz_chart("""
    digraph pipeline {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor=lightblue, fontname="Arial"];
        edge [fontname="Arial"];

        "Data Collection\n(IEEE-CIS, PaySim)" -> "Time-Ordered Split\n(70/15/15)";
        "Time-Ordered Split\n(70/15/15)" -> "Baseline Models\n(LR, RF, XGB, LGBM)";
        "Baseline Models\n(LR, RF, XGB, LGBM)" -> "Delayed Labels Sim\n(1/7/30 days)";
        "Delayed Labels Sim\n(1/7/30 days)" -> "Drift Monitoring\n(PSI/KS on features & scores)";
        "Drift Monitoring\n(PSI/KS on features & scores)" -> "SHAP Explanations\n(Per-alert + Global)";
        "SHAP Explanations\n(Per-alert + Global)" -> "Retraining Strategies\n(Static/Scheduled/Drift)";
        "Retraining Strategies\n(Static/Scheduled/Drift)" -> "Operational Evaluation\n(PR-AUC, P@K, FP/100k)";
    }
    """)

# ============================================================
# PAGE: RISK SCORING
# ============================================================
elif page == "🎯 Risk Scoring":
    st.title("🎯 Real-time Risk Scoring")

    # ── Model Upload ──
    st.subheader("📤 Upload Custom Model")
    uploaded_model = st.file_uploader(
        "Upload a trained model (.joblib)",
        type=["joblib"],
        key="risk_model_upload"
    )
    if uploaded_model is not None:
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp.write(uploaded_model.read())
                tmp_path = tmp.name
            uploaded_model_obj = joblib.load(tmp_path)
            meta_path = tmp_path.replace(".joblib", ".json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    model_meta = json.load(f)
            model = uploaded_model_obj
            st.success(f"✅ Model uploaded: {uploaded_model.name}")
            if model_meta:
                st.info(f"Model metrics: {model_meta.get('val_metrics', {})}")
        except Exception as exc:
            st.error(f"❌ Failed to load model: {exc}")

    st.divider()

    if model is None:
        st.warning("Model not loaded. Please train models first using notebook 02_baselines.ipynb")
    else:
        # Load test data for scoring
        try:
            test_ieee = pd.read_parquet(DATA_ROOT / 'ieee-cis' / 'splits' / 'test.parquet')
            test_ieee = apply_fe(test_ieee, 'ieee-cis')
            X_test, y_test = prepare_xy(test_ieee)

            # Get scores
            scores = model.predict_proba(X_test)[:, 1]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Risk Score Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=scores[y_test == 0], name='Legitimate',
                    opacity=0.7, nbinsx=50, marker_color='green'
                ))
                fig.add_trace(go.Histogram(
                    x=scores[y_test == 1], name='Fraud',
                    opacity=0.7, nbinsx=50, marker_color='red'
                ))
                fig.update_layout(barmode='overlay', xaxis_title='Risk Score', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Score Statistics")
                st.write(f"**Mean Score**: {scores.mean():.4f}")
                st.write(f"**Std Score**: {scores.std():.4f}")
                st.write(f"**Max Score**: {scores.max():.4f}")
                st.write(f"**Fraud Rate**: {y_test.mean():.4%}")

                # Threshold analysis
                st.subheader("Threshold Analysis")
                threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
                y_pred = (scores >= threshold).astype(int)
                tp = ((y_pred == 1) & (y_test == 1)).sum()
                fp = ((y_pred == 1) & (y_test == 0)).sum()
                fn = ((y_pred == 0) & (y_test == 1)).sum()
                tn = ((y_pred == 0) & (y_test == 0)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                st.write(f"**Precision**: {precision:.4f}")
                st.write(f"**Recall**: {recall:.4f}")
                st.write(f"**F1**: {f1:.4f}")
                st.write(f"**Alerts**: {tp + fp:,} / {len(y_test):,}")

                # Save scoring session to prediction history
                st.session_state.prediction_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_meta.get("model", "uploaded"),
                    "dataset": "ieee-cis",
                    "n_predictions": int(len(scores)),
                    "fraud_rate": float(y_test.mean()),
                    "avg_score": float(scores.mean()),
                    "max_score": float(scores.max()),
                    "min_score": float(scores.min()),
                    "threshold": float(threshold),
                    "predicted_frauds": int(y_pred.sum()),
                    "actual_frauds": int(y_test.sum()),
                })

        except Exception as e:
            st.error(f"Error loading test data: {e}")

# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    st.markdown("Compare baseline model performance across IEEE-CIS and PaySim datasets.")

    # Load metrics from reports
    ieee_df = reports.get('ieee_test_metrics', pd.DataFrame())
    paysim_df = reports.get('paysim_test_metrics', pd.DataFrame())

    if ieee_df.empty and paysim_df.empty:
        st.warning("No model metrics found. Run notebook 02_baselines.ipynb to generate evaluation results.")
    else:
        # Normalize column names
        def normalize_metrics(df):
            if df.empty:
                return df
            df = df.copy()
            # Try to find model column
            model_col = None
            for col in df.columns:
                if col.lower() in ('model', 'algorithm', 'name'):
                    model_col = col
                    break
            if model_col and model_col != 'model':
                df = df.rename(columns={model_col: 'model'})
            return df

        ieee_df = normalize_metrics(ieee_df)
        paysim_df = normalize_metrics(paysim_df)

        # Combined comparison table
        st.subheader("📊 Side-by-Side Comparison")

        metrics_to_show = ['pr_auc', 'precision_at_100', 'recall_at_100', 'fp_per_100k_at_budget_100']
        available_metrics = [m for m in metrics_to_show if m in ieee_df.columns or m in paysim_df.columns]

        if 'model' in ieee_df.columns or 'model' in paysim_df.columns:
            # Build comparison from both datasets
            all_models = set()
            if 'model' in ieee_df.columns:
                all_models.update(ieee_df['model'].tolist())
            if 'model' in paysim_df.columns:
                all_models.update(paysim_df['model'].tolist())

            comparison_rows = []
            for model in sorted(all_models):
                row = {'model': model}
                for m in available_metrics:
                    ieee_val = None
                    paysim_val = None
                    if 'model' in ieee_df.columns and m in ieee_df.columns:
                        match = ieee_df[ieee_df['model'] == model]
                        if not match.empty:
                            ieee_val = match.iloc[0][m]
                    if 'model' in paysim_df.columns and m in paysim_df.columns:
                        match = paysim_df[paysim_df['model'] == model]
                        if not match.empty:
                            paysim_val = match.iloc[0][m]
                    row[f'ieee_{m}'] = ieee_val
                    row[f'paysim_{m}'] = paysim_val
                comparison_rows.append(row)

            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Best model by PR-AUC (IEEE-CIS)
            if 'ieee_pr_auc' in comp_df.columns:
                best_model = comp_df.loc[comp_df['ieee_pr_auc'].idxmax()]
                st.success(f"🏆 Best model (IEEE-CIS PR-AUC): **{best_model['model']}** with PR-AUC = {best_model['ieee_pr_auc']:.4f}")

            # Charts
            st.subheader("📈 Performance by Model")

            col1, col2 = st.columns(2)

            with col1:
                if 'ieee_pr_auc' in comp_df.columns and comp_df['ieee_pr_auc'].notna().any():
                    fig = px.bar(
                        comp_df.dropna(subset=['ieee_pr_auc']).sort_values('ieee_pr_auc', ascending=True),
                        x='ieee_pr_auc', y='model', orientation='h',
                        title='PR-AUC by Model (IEEE-CIS)',
                        labels={'ieee_pr_auc': 'PR-AUC', 'model': 'Model'},
                        color='ieee_pr_auc', color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'ieee_precision_at_100' in comp_df.columns and comp_df['ieee_precision_at_100'].notna().any():
                    fig = px.bar(
                        comp_df.dropna(subset=['ieee_precision_at_100']).sort_values('ieee_precision_at_100', ascending=True),
                        x='ieee_precision_at_100', y='model', orientation='h',
                        title='Precision@100 by Model (IEEE-CIS)',
                        labels={'ieee_precision_at_100': 'Precision@100', 'model': 'Model'},
                        color='ieee_precision_at_100', color_continuous_scale='Plasma'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Model column not found in metrics. Available columns: " + ", ".join(
                list(ieee_df.columns) + list(paysim_df.columns)))

# ============================================================
# PAGE: PIPELINE CONTROL
# ============================================================
elif page == "🚀 Pipeline Control":
    st.title("🚀 Pipeline Control")

    st.markdown("Run pipeline operations directly from the dashboard. "
                "Operations call the backend pipeline modules in real time.")

    dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="pc_dataset")

    # ── Train ──
    st.subheader("📚 Train Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        pc_model = st.selectbox("Model",
                                ["lightgbm", "logistic_regression", "random_forest", "xgboost", "catboost", "all"],
                                key="pc_model")
    with col2:
        pc_train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.7, 0.05, key="pc_train_ratio")
    with col3:
        pc_val_ratio = st.slider("Val Ratio", 0.05, 0.3, 0.15, 0.05, key="pc_val_ratio")
    pc_test_ratio = st.slider("Test Ratio", 0.05, 0.3, 0.15, 0.05, key="pc_test_ratio")

    # Ensemble options
    pc_create_ensemble = st.checkbox("Create Ensemble from Top-K Models", key="pc_create_ensemble")
    pc_top_k = st.slider("Top-K for Ensemble", 2, 5, 3, key="pc_top_k")

    if st.button("🏋️ Train", key="pc_train_btn"):
        with st.spinner("Training models..."):
            try:
                result = stage_train(
                    dataset=dataset,
                    model_name=None if pc_model == "all" else pc_model,
                    train_ratio=pc_train_ratio,
                    val_ratio=pc_val_ratio,
                    test_ratio=pc_test_ratio,
                    quiet=True,
                    create_ensemble=pc_create_ensemble,
                    top_k=pc_top_k,
                )
                st.success("✅ Training completed!")
                if "models" in result:
                    for name, m in result["models"].items():
                        if "error" in m:
                            st.error(f"{name}: {m['error']}")
                        else:
                            st.write(f"**{name}**: Val PR-AUC = {m['val_metrics']['pr_auc']:.4f}, "
                                     f"Test PR-AUC = {m['test_metrics']['pr_auc']:.4f}")
                if "ensemble" in result and result["ensemble"]:
                    ens = result["ensemble"]
                    st.write(f"**Ensemble**: Test PR-AUC = {ens.get('test_metrics', {}).get('pr_auc', 'N/A'):.4f}")
                    st.write(f"  Models: {ens.get('ensemble_models', [])}")
                    st.write(f"  Weights: {ens.get('ensemble_weights', {})}")
            except Exception as exc:
                st.error(f"❌ Training failed: {exc}")

    st.divider()

    # ── Drift ──
    st.subheader("📈 Drift Detection")
    col1, col2 = st.columns(2)
    with col1:
        pc_ref_days = st.slider("Reference Days", 1, 30, 7, key="pc_ref_days")
    with col2:
        pc_cur_days = st.slider("Current Days", 1, 30, 7, key="pc_cur_days")

    if st.button("🔍 Detect Drift", key="pc_drift_btn"):
        with st.spinner("Running drift detection..."):
            try:
                result = stage_drift(
                    dataset=dataset,
                    reference_window_days=pc_ref_days,
                    current_window_days=pc_cur_days,
                )
                st.success("✅ Drift detection completed!")
                st.write(f"**Alerts**: {len(result.get('alerts', []))}")
                for alert in result.get("alerts", []):
                    st.write(f"- {alert['type']}: {alert['column']} ({alert['metric']}={alert['value']:.4f})")
            except Exception as exc:
                st.error(f"❌ Drift detection failed: {exc}")

    st.divider()

    # ── Retrain ──
    st.subheader("🔄 Drift-Triggered Retraining")
    col1, col2 = st.columns(2)
    with col1:
        pc_retrain_model = st.selectbox("Model",
                                        ["lightgbm", "logistic_regression", "random_forest", "xgboost"],
                                        key="pc_retrain_model")
    with col2:
        pc_force = st.checkbox("Force Retrain", key="pc_force")
    pc_psi = st.slider("PSI Threshold", 0.1, 0.5, 0.2, 0.05, key="pc_psi")
    pc_ks = st.slider("KS Threshold", 0.01, 0.2, 0.05, 0.01, key="pc_ks")

    if st.button("🔄 Retrain", key="pc_retrain_btn"):
        with st.spinner("Checking drift and retraining if needed..."):
            try:
                result = stage_retrain(
                    dataset=dataset,
                    model_name=pc_retrain_model,
                    train_ratio=pc_train_ratio,
                    val_ratio=pc_val_ratio,
                    test_ratio=pc_test_ratio,
                    psi_threshold=pc_psi,
                    ks_threshold=pc_ks,
                    force=pc_force,
                    quiet=True,
                )
                if result["retrained"]:
                    st.success(f"✅ Model retrained (trigger: {result['trigger']})!")
                else:
                    st.info("ℹ️ No retraining needed — drift below thresholds.")
                st.write(f"**Val PR-AUC**: {result['val_metrics']['pr_auc']:.4f}")
                st.write(f"**Test PR-AUC**: {result['test_metrics']['pr_auc']:.4f}")
                if result["drift_alerts"]:
                    st.write(f"**Drift alerts**: {len(result['drift_alerts'])}")
            except Exception as exc:
                st.error(f"❌ Retraining failed: {exc}")

    st.divider()

    # ── River Drift Detection ──
    st.subheader("🌊 River Streaming Drift Detection")
    st.markdown("Real-time concept drift detection using ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN.")

    col1, col2 = st.columns(2)
    with col1:
        pc_river_detectors = st.multiselect(
            "River Detectors",
            ["adwin", "ddm", "eddm", "hddm_a", "hddm_w", "kswin"],
            default=["adwin", "ddm", "eddm", "hddm_a", "hddm_w", "kswin"],
            key="pc_river_detectors"
        )
    with col2:
        pc_river_ref_size = st.slider("Reference Window Size", 100, 5000, 1000, 100, key="pc_river_ref_size")

    pc_river_psi = st.slider("PSI Threshold", 0.1, 0.5, 0.2, 0.05, key="pc_river_psi")
    pc_river_ks = st.slider("KS Threshold", 0.01, 0.2, 0.05, 0.01, key="pc_river_ks")
    pc_river_score_threshold = st.slider("Score Threshold", 0.1, 0.9, 0.5, 0.05, key="pc_river_score_threshold")

    if st.button("🌊 Run River Drift Detection", key="pc_river_drift_btn"):
        with st.spinner("Running River drift detection..."):
            try:
                from src.pipeline import stage_river_drift
                result = stage_river_drift(
                    dataset=dataset,
                    reference_window_size=pc_river_ref_size,
                    psi_threshold=pc_river_psi,
                    ks_threshold=pc_river_ks,
                    score_threshold=pc_river_score_threshold,
                    river_detectors=pc_river_detectors,
                    quiet=True,
                )
                st.success("✅ River drift detection completed!")
                st.write(f"**Total alerts**: {len(result.get('alerts', []))}")
                st.write(f"**Detectors used**: {list(result.get('river_detectors', {}).keys())}")
                for detector, detected in result.get('river_detectors', {}).items():
                    st.write(f"  - {detector}: {'🔴 Drift' if detected else '🟢 No drift'}")
            except Exception as exc:
                st.error(f"❌ River drift detection failed: {exc}")

    st.divider()

    # ── Online Learning ──
    st.subheader("🔁 Online/Incremental Learning")
    st.markdown("Train models incrementally on streaming data without full retraining.")

    col1, col2 = st.columns(2)
    with col1:
        pc_online_model = st.selectbox(
            "Online Model Type",
            ["logistic_regression", "adaptive_random_forest", "hoeffding_tree"],
            key="pc_online_model"
        )
    with col2:
        pc_online_buffer = st.slider("Buffer Size", 100, 5000, 1000, 100, key="pc_online_buffer")

    pc_online_eval_interval = st.slider("Evaluation Interval", 100, 2000, 500, 100, key="pc_online_eval_interval")

    if st.button("🔁 Run Online Learning", key="pc_online_btn"):
        with st.spinner("Running online learning..."):
            try:
                from src.pipeline import stage_online_learning
                result = stage_online_learning(
                    dataset=dataset,
                    model_type=pc_online_model,
                    train_ratio=pc_train_ratio,
                    val_ratio=pc_val_ratio,
                    test_ratio=pc_test_ratio,
                    buffer_size=pc_online_buffer,
                    eval_interval=pc_online_eval_interval,
                    save=True,
                    quiet=True,
                )
                st.success("✅ Online learning completed!")
                st.write(f"**Model**: {result.get('model_type')}")
                st.write(f"**Train samples**: {result.get('train_samples')}")
                st.write(f"**Test samples**: {result.get('test_samples')}")
                st.write(f"**Val PR-AUC**: {result.get('val_metrics', {}).get('pr_auc', 'N/A'):.4f}")
                st.write(f"**Test PR-AUC**: {result.get('test_metrics', {}).get('pr_auc', 'N/A'):.4f}")
                st.write(f"**Drift alerts**: {len(result.get('drift_alerts', []))}")
            except Exception as exc:
                st.error(f"❌ Online learning failed: {exc}")

    st.divider()

    # ── Run all ──
    st.subheader("🔄 Full Pipeline")
    st.markdown("Run train → evaluate → drift → retrain → shap in sequence.")

    if st.button("▶️ Run All", key="pc_run_all_btn"):
        with st.spinner("Running full pipeline..."):
            try:
                stage_train(
                    dataset=dataset,
                    model_name=pc_retrain_model,
                    train_ratio=pc_train_ratio,
                    val_ratio=pc_val_ratio,
                    test_ratio=pc_test_ratio,
                    save=True,
                    quiet=True,
                )
                st.success("✅ Train complete")
                stage_drift(dataset=dataset)
                st.success("✅ Drift detection complete")
                stage_retrain(
                    dataset=dataset,
                    model_name=pc_retrain_model,
                    train_ratio=pc_train_ratio,
                    val_ratio=pc_val_ratio,
                    test_ratio=pc_test_ratio,
                    psi_threshold=pc_psi,
                    ks_threshold=pc_ks,
                    force=pc_force,
                    quiet=True,
                )
                st.success("✅ Retrain check complete")
                st.success("🎉 Full pipeline completed!")
            except Exception as exc:
                st.error(f"❌ Pipeline failed: {exc}")

# ============================================================
# PAGE: SHAP EXPLANATIONS
# ============================================================
elif page == "🔍 SHAP Explanations":
    st.title("🔍 SHAP Explanations")

    if model is None:
        st.warning("Model not loaded. Please train models first.")
    else:
        # Load test data
        try:
            test_ieee = pd.read_parquet(DATA_ROOT / 'ieee-cis' / 'splits' / 'test.parquet')
            test_ieee = apply_fe(test_ieee, 'ieee-cis')
            X_test, y_test = prepare_xy(test_ieee)

            # Load training data for background
            train_ieee = pd.read_parquet(DATA_ROOT / 'ieee-cis' / 'splits' / 'train.parquet')
            train_ieee = apply_fe(train_ieee, 'ieee-cis')
            X_train, _ = prepare_xy(train_ieee)

            # Initialize SHAP explainer
            with st.spinner("Initializing SHAP explainer..."):
                explainer = SHAPExplainer(model, X_train.sample(n=min(500, len(X_train)), random_state=42),
                                          feature_names=X_train.columns.tolist(), max_background=500)

            tab1, tab2, tab3 = st.tabs(["🔬 Per-Instance", "📊 Global Importance", "📈 Importance Over Time"])

            with tab1:
                st.subheader("Per-Instance SHAP Explanation")

                # Select transaction
                scores = model.predict_proba(X_test)[:, 1]
                top_indices = np.argsort(scores)[-20:]

                selected_idx = st.selectbox(
                    "Select Transaction (Top 20 Riskiest)",
                    top_indices[::-1],
                    format_func=lambda x: f"Idx {x} - Score: {scores[x]:.4f} - Label: {'Fraud' if y_test.iloc[x]==1 else 'Legit'}"
                )

                if selected_idx is not None:
                    x_row = X_test.loc[selected_idx].to_frame().T
                    explanation = explainer.explain_instance(x_row, max_display=15)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write(f"**Prediction**: {explanation['prediction']:.4f}")
                        st.write(f"**Base Value**: {explanation['base_value']:.4f}")
                        st.write(f"**True Label**: {'Fraud' if y_test.iloc[selected_idx]==1 else 'Legitimate'}")

                    with col2:
                        # Waterfall chart
                        contrib = explanation['feature_contributions'][:15]
                        feat_names = [c[0] for c in contrib]
                        feat_vals = [c[1] for c in contrib]

                        fig = go.Figure(go.Waterfall(
                            name="SHAP",
                            orientation="h",
                            measure=["relative"] * len(feat_vals) + ["total"],
                            x=feat_vals + [sum(feat_vals)],
                            y=feat_names + ["Prediction"],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                        ))
                        fig.update_layout(height=500, title="SHAP Waterfall Plot")
                        st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Global Feature Importance (Mean |SHAP|)")

                with st.spinner("Computing global importance..."):
                    importance_df = explainer.global_importance(X_test, max_samples=min(2000, len(X_test)))

                top_n = st.slider("Top N Features", 5, 50, 20)

                fig = px.bar(
                    importance_df.head(top_n).iloc[::-1],
                    x='mean_abs_shap', y='feature',
                    orientation='h', error_x='std_shap',
                    title=f'Top {top_n} Features by Mean |SHAP|',
                    labels={'mean_abs_shap': 'Mean |SHAP Value|', 'feature': 'Feature'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Summary plot info
                st.info("💡 For beeswarm summary plot, run notebook 05_shap_explanations.ipynb")

            with tab3:
                st.subheader("Feature Importance Stability Over Time")

                if not reports['shap_stability'].empty:
                    stability_df = reports['shap_stability']

                    # Plot CV for top features
                    top_features = stability_df.head(20)

                    fig = px.bar(
                        top_features,
                        x='feature', y='cv',
                        color='cv', color_continuous_scale='RdYlGn_r',
                        title='SHAP Importance Coefficient of Variation (Lower = More Stable)',
                        labels={'cv': 'Coefficient of Variation', 'feature': 'Feature'}
                    )
                    fig.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Stable threshold")
                    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Unstable threshold")
                    fig.update_layout(xaxis_tickangle=-45, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Stability table
                    st.dataframe(stability_df[['feature', 'mean_importance', 'std_importance', 'cv', 'stability']])
                else:
                    st.info("Run notebook 05_shap_explanations.ipynb to generate stability data")

        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# PAGE: DRIFT MONITORING
# ============================================================
elif page == "📈 Drift Monitoring":
    st.title("📈 Drift Monitoring (PSI/KS)")

    # ── Live Drift Detection ──
    st.subheader("🔴 Live Drift Detection")
    st.markdown("Run PSI/KS drift detection on live data windows.")
    col1, col2, col3 = st.columns(3)
    with col1:
        live_dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="live_dataset")
    with col2:
        live_ref_days = st.slider("Reference Days", 1, 30, 7, key="live_ref_days")
    with col3:
        live_cur_days = st.slider("Current Days", 1, 30, 7, key="live_cur_days")

    if st.button("🔍 Run Drift Detection", key="live_drift_btn"):
        with st.spinner("Running drift detection..."):
            try:
                result = stage_drift(
                    dataset=live_dataset,
                    reference_window_days=live_ref_days,
                    current_window_days=live_cur_days,
                )
                st.session_state["live_drift_result"] = result
                st.success("✅ Drift detection completed!")
            except Exception as exc:
                st.error(f"❌ Drift detection failed: {exc}")

    # Show live results
    if "live_drift_result" in st.session_state:
        result = st.session_state["live_drift_result"]
        alerts = result.get("alerts", [])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", len(alerts))
        with col2:
            feature_count = sum(1 for a in alerts if a.get("type") == "feature_drift")
            st.metric("Feature Drift", feature_count)
        with col3:
            score_count = sum(1 for a in alerts if "score_drift" in a.get("type", ""))
            st.metric("Score Drift", score_count)

        # Status summary
        col1, col2, col3, col4 = st.columns(4)
        pending_live = sum(1 for a in alerts if st.session_state.alert_status.get(alert_key(a), "pending") == "pending")
        acked_live = sum(1 for a in alerts if st.session_state.alert_status.get(alert_key(a), "pending") == "acknowledged")
        resolved_live = sum(1 for a in alerts if st.session_state.alert_status.get(alert_key(a), "pending") == "resolved")
        with col1:
            st.metric("Pending", pending_live)
        with col2:
            st.metric("Acknowledged", acked_live)
        with col3:
            st.metric("Resolved", resolved_live)

        if alerts:
            st.subheader("🚨 Active Drift Alerts")
            for alert in alerts:
                value = alert.get("value", 0)
                if value > 0.3:
                    severity = "🔴 High"
                elif value > 0.2:
                    severity = "🟡 Medium"
                else:
                    severity = "🟢 Low"
                key = alert_key(alert)
                current_status = st.session_state.alert_status.get(key, "pending")

                col_a, col_b, col_c = st.columns([5, 1, 1])
                with col_a:
                    status_icon = {"pending": "⏳", "acknowledged": "✓", "resolved": "✅"}[current_status]
                    st.write(f"{status_icon} {severity} — {alert['type']}: {alert['column']} ({alert['metric']}={value:.4f})")
                with col_b:
                    if current_status != "acknowledged":
                        if st.button("✓ Ack", key=f"ack_live_{key}"):
                            st.session_state.alert_status[key] = "acknowledged"
                            save_alert_status(st.session_state.alert_status)
                            st.rerun()
                with col_c:
                    if current_status != "resolved":
                        if st.button("✅ Resolve", key=f"res_live_{key}"):
                            st.session_state.alert_status[key] = "resolved"
                            save_alert_status(st.session_state.alert_status)
                            st.rerun()

    st.divider()

    # Existing reports
    if not reports['drift_alerts'].empty:
        alerts_df = reports['drift_alerts']

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", len(alerts_df))
        with col2:
            feature_alerts = alerts_df[alerts_df['type'] == 'feature_drift'] if 'type' in alerts_df.columns else pd.DataFrame()
            st.metric("Feature Drift Alerts", len(feature_alerts))
        with col3:
            score_alerts = alerts_df[alerts_df['type'] == 'score_drift'] if 'type' in alerts_df.columns else pd.DataFrame()
            st.metric("Score Drift Alerts", len(score_alerts))

        # Alerts table with management
        st.subheader("📋 Drift Alerts Log")
        if not alerts_df.empty:
            # Status filter
            col1, col2 = st.columns([1, 3])
            with col1:
                filter_status = st.selectbox(
                    "Filter by status",
                    ["All", "Pending", "Acknowledged", "Resolved"],
                    key="drift_status_filter"
                )

            # Status summary
            col1, col2, col3, col4 = st.columns(4)
            pending_stored = sum(1 for _, r in alerts_df.iterrows() if st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == "pending")
            acked_stored = sum(1 for _, r in alerts_df.iterrows() if st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == "acknowledged")
            resolved_stored = sum(1 for _, r in alerts_df.iterrows() if st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == "resolved")
            with col1:
                st.metric("Pending", pending_stored)
            with col2:
                st.metric("Acknowledged", acked_stored)
            with col3:
                st.metric("Resolved", resolved_stored)

            # Per-row alerts with actions
            for idx, (_, row) in enumerate(alerts_df.iterrows()):
                alert = row.to_dict()
                value = alert.get("value", 0)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0.0
                if value > 0.3:
                    severity = "🔴 High"
                elif value > 0.2:
                    severity = "🟡 Medium"
                else:
                    severity = "🟢 Low"
                key = alert_key(alert)
                current_status = st.session_state.alert_status.get(key, "pending")

                if filter_status != "All" and current_status.title() != filter_status:
                    continue

                col_a, col_b, col_c = st.columns([5, 1, 1])
                with col_a:
                    status_icon = {"pending": "⏳", "acknowledged": "✓", "resolved": "✅"}[current_status]
                    st.write(f"{status_icon} {severity} — {alert.get('type','')}: {alert.get('column','')} ({alert.get('metric','')}={value:.4f})")
                with col_b:
                    if current_status != "acknowledged":
                        if st.button("✓ Ack", key=f"ack_stored_{idx}_{key}"):
                            st.session_state.alert_status[key] = "acknowledged"
                            save_alert_status(st.session_state.alert_status)
                            st.rerun()
                with col_c:
                    if current_status != "resolved":
                        if st.button("✅ Resolve", key=f"res_stored_{idx}_{key}"):
                            st.session_state.alert_status[key] = "resolved"
                            save_alert_status(st.session_state.alert_status)
                            st.rerun()

            # Alert type breakdown
            if 'type' in alerts_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Alert Type Distribution")
                    type_counts = alerts_df['type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Alert Types")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if 'column' in alerts_df.columns:
                        st.subheader("Most Alerted Features")
                        feat_counts = alerts_df['column'].value_counts().head(10)
                        fig = px.bar(x=feat_counts.values, y=feat_counts.index, orientation='h',
                                   labels={'x': 'Alert Count', 'y': 'Feature'})
                        st.plotly_chart(fig, use_container_width=True)

        # PSI over time visualization
        st.subheader("📊 PSI Trends Over Time")
        st.info("Run notebook 04_drift_monitoring.ipynb for detailed PSI/KS time series plots")

    else:
        st.warning("No drift alerts found. Run notebook 04_drift_monitoring.ipynb first.")
        st.code("jupyter notebook notebooks/04_drift_monitoring.ipynb")

# ============================================================
# PAGE: RETRAINING STRATEGIES
# ============================================================
elif page == "⚙️ Retraining Strategies":
    st.title("⚙️ Retraining Strategy Comparison")

    if not reports['retrain_comparison'].empty:
        comp_df = reports['retrain_comparison']

        st.subheader("Strategy Performance Comparison")
        st.dataframe(comp_df, use_container_width=True)

        # Metrics comparison
        metrics_to_plot = ['pr_auc', 'precision_at_100', 'recall_at_100', 'fp_per_100k_at_budget_100']
        available_metrics = [m for m in metrics_to_plot if m in comp_df.columns]

        if available_metrics:
            st.subheader("Operational Metrics by Strategy")

            for metric in available_metrics:
                fig = px.bar(
                    comp_df, x='strategy', y=metric,
                    title=metric.replace('_', ' ').title(),
                    color='strategy', text_auto='.4f'
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Detailed results
        if not reports.get('retrain_detailed', pd.DataFrame()).empty:
            detailed = reports['retrain_detailed']
            st.subheader("Detailed Window-by-Window Results")
            st.dataframe(detailed, use_container_width=True)

    else:
        st.warning("No retraining results found. Run notebook 06_retraining_strategies.ipynb first.")
        st.code("jupyter notebook notebooks/06_retraining_strategies.ipynb")

    # Strategy descriptions
    with st.expander("📖 Strategy Descriptions"):
        st.markdown("""
        ### Static
        - Never retrain after initial training
        - Baseline for comparison

        ### Scheduled (Weekly)
        - Retrain every 7 days regardless of data changes
        - Simple but may retrain unnecessarily

        ### Drift-Triggered
        - Retrain only when PSI > 0.2 or KS > 0.05 detected
        - Force retrain after 90 days max model age
        - **Expected: 30% fewer retrains than scheduled with comparable performance**
        """)

# ============================================================
# PAGE: DELAYED LABELS
# ============================================================
elif page == "⏱️ Delayed Labels":
    st.title("⏱️ Delayed Labels Impact Analysis")

    if not reports['delay_impact'].empty:
        delay_df = reports['delay_impact']

        st.subheader("Performance Degradation with Label Delay")
        st.dataframe(delay_df, use_container_width=True)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=delay_df['delay_days'], y=delay_df['avg_pr_auc'],
            mode='lines+markers', name='PR-AUC', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=delay_df['delay_days'], y=delay_df['avg_p100'],
            mode='lines+markers', name='P@100', line=dict(color='red'),
            yaxis='y2'
        ))
        fig.update_layout(
            title='Performance vs Label Delay',
            xaxis_title='Label Delay (Days)',
            yaxis=dict(title='PR-AUC', side='left'),
            yaxis2=dict(title='P@100', side='right', overlaying='y'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Degradation summary
        st.subheader("Relative Degradation vs Oracle (0-day delay)")
        oracle_pr = delay_df[delay_df['delay_days'] == 0]['avg_pr_auc'].values[0] if 0 in delay_df['delay_days'].values else delay_df['avg_pr_auc'].max()

        for _, row in delay_df.iterrows():
            if row['delay_days'] > 0:
                degradation = (oracle_pr - row['avg_pr_auc']) / oracle_pr * 100
                st.write(f"**{row['delay_days']}-day delay**: PR-AUC degradation = {degradation:.1f}%")

    else:
        st.warning("No delayed labels results found. Run notebook 03_delayed_labels.ipynb first.")
        st.code("jupyter notebook notebooks/03_delayed_labels.ipynb")

# ============================================================
# PAGE: PREDICTION HISTORY
# ============================================================
elif page == "📜 Prediction History":
    st.title("📜 Prediction History")

    history = st.session_state.prediction_history

    if not history:
        st.info("No prediction sessions yet. Go to Risk Scoring to generate predictions.")
    else:
        # Convert to DataFrame
        hist_df = pd.DataFrame(history)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", len(hist_df))
        with col2:
            st.metric("Total Predictions", int(hist_df['n_predictions'].sum()))
        with col3:
            st.metric("Avg Fraud Rate", f"{hist_df['fraud_rate'].mean():.2%}")
        with col4:
            st.metric("Avg Risk Score", f"{hist_df['avg_score'].mean():.4f}")

        st.divider()

        # Filters
        st.subheader("🔍 Filter History")
        col1, col2 = st.columns(2)
        with col1:
            model_filter = st.selectbox(
                "Model",
                ["All"] + sorted(hist_df['model'].unique().tolist()),
                key="ph_model_filter"
            )
        with col2:
            date_filter = st.date_input(
                "From date",
                value=None,
                key="ph_date_filter"
            )

        filtered_df = hist_df.copy()
        if model_filter != "All":
            filtered_df = filtered_df[filtered_df['model'] == model_filter]
        if date_filter:
            filtered_df = filtered_df[
                pd.to_datetime(filtered_df['timestamp']).dt.date >= date_filter
            ]

        # History table
        st.subheader("📋 Prediction Sessions")
        display_cols = ['timestamp', 'model', 'dataset', 'n_predictions', 'fraud_rate',
                        'avg_score', 'max_score', 'threshold', 'predicted_frauds', 'actual_frauds']
        display_df = filtered_df[display_cols].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['fraud_rate'] = display_df['fraud_rate'].apply(lambda x: f"{x:.2%}")
        display_df['avg_score'] = display_df['avg_score'].apply(lambda x: f"{x:.4f}")
        display_df['max_score'] = display_df['max_score'].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp"),
                "model": st.column_config.TextColumn("Model"),
                "dataset": st.column_config.TextColumn("Dataset"),
                "n_predictions": st.column_config.NumberColumn("Predictions", format="%d"),
                "fraud_rate": st.column_config.TextColumn("Actual Fraud Rate"),
                "avg_score": st.column_config.TextColumn("Avg Score"),
                "max_score": st.column_config.TextColumn("Max Score"),
                "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                "predicted_frauds": st.column_config.NumberColumn("Predicted Frauds", format="%d"),
                "actual_frauds": st.column_config.NumberColumn("Actual Frauds", format="%d"),
            }
        )

        # Score trend chart
        st.subheader("📈 Risk Score Trend")
        if len(filtered_df) > 1:
            trend_df = filtered_df.copy()
            trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'])
            trend_df = trend_df.sort_values('timestamp')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df['timestamp'], y=trend_df['avg_score'],
                mode='lines+markers', name='Avg Risk Score',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=trend_df['timestamp'], y=trend_df['max_score'],
                mode='lines+markers', name='Max Risk Score',
                line=dict(color='red')
            ))
            fig.update_layout(
                title='Average and Max Risk Score Over Sessions',
                xaxis_title='Session Time',
                yaxis_title='Risk Score',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run more scoring sessions to see the trend chart.")

        # Actions
        st.divider()
        st.subheader("⚙️ Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", key="ph_clear"):
                st.session_state.prediction_history = []
                st.success("Prediction history cleared.")
                st.rerun()
        with col2:
            csv = hist_df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv,
                file_name="prediction_history.csv",
                mime="text/csv",
                key="ph_export"
            )

# ============================================================
# PAGE: REPORTS
# ============================================================
elif page == "📋 Reports":
    st.title("📋 Generated Reports")

    report_files = {
        'Final Evaluation Report': REPORTS_DIR / 'final_evaluation_report.txt',
        'Drift Alerts Log': REPORTS_DIR / 'drift_alerts_log.csv',
        'SHAP Global Importance': REPORTS_DIR / 'shap_global_importance.csv',
        'SHAP Stability': REPORTS_DIR / 'shap_stability.csv',
        'Retraining Comparison': REPORTS_DIR / 'retraining_strategy_comparison.csv',
        'Delayed Labels Impact': REPORTS_DIR / 'delay_impact_summary.csv',
        'Baseline Test Metrics (IEEE)': REPORTS_DIR / 'ieee_test_metrics.csv',
        'Baseline Test Metrics (PaySim)': REPORTS_DIR / 'paysim_test_metrics.csv',
    }

    for name, path in report_files.items():
        with st.expander(f"📄 {name}"):
            if path.exists():
                if path.suffix == '.txt':
                    with open(path) as f:
                        st.text(f.read())
                else:
                    df = pd.read_csv(path)
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        f"Download {name}",
                        df.to_csv(index=False),
                        file_name=path.name,
                        mime='text/csv'
                    )
            else:
                st.warning(f"File not found: {path}")

    # Generate final report button
    if st.button("🔄 Regenerate Final Report"):
        with st.spinner("Generating report..."):
            # This would call the evaluation module
            st.success("Report regeneration triggered. Check notebook 07_evaluation.ipynb")

# ============================================================
# PAGE: API INTEGRATION
# ============================================================
elif page == "🔌 API Integration":
    st.title("🔌 API Integration")
    st.markdown("Interact with the UPI Fraud Detection API endpoints directly from the dashboard.")

    # Health check
    st.subheader("🏥 Health Check")
    if st.button("Check API Health", key="api_health_btn"):
        try:
            response = api_client.get("/health")
            if response.status_code == 200:
                st.success(f"✅ API is healthy: {response.json()}")
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except Exception as exc:
            st.error(f"❌ API health check failed: {exc}")

    st.divider()

    # Prediction endpoint
    st.subheader("🔮 Test Prediction")
    st.markdown("Send a prediction request to the API.")

    with st.form("api_predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            api_dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="api_dataset")
            api_model = st.selectbox("Model", ["lightgbm", "logistic_regression", "random_forest", "xgboost"], key="api_model")
        with col2:
            api_features_json = st.text_area(
                "Features (JSON)",
                value='{"TransactionAmt": 150.0, "card1": 12345, "card4": "visa", "ProductCD": "W"}',
                height=150,
                key="api_features_json"
            )

        submitted = st.form_submit_button("Send Prediction Request")

    if submitted:
        try:
            import json as _json
            features = _json.loads(api_features_json)
            response = api_client.post("/predict", json={
                "dataset": api_dataset,
                "model": api_model,
                "features": features,
            })
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Prediction successful!")
                preds = data.get("predictions", [])[:3]
                if preds:
                    cols = st.columns(min(len(preds), 3))
                    for i, pred in enumerate(preds):
                        with cols[i]:
                            st.metric(f"Prediction {i + 1}", f"{pred['fraud_score']:.4f}",
                                     delta="Fraud" if pred["is_fraud"] == 1 else "Legit")
                st.json(data)
            else:
                st.error(f"❌ API returned status {response.status_code}: {response.json()}")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON in features field")
        except Exception as exc:
            st.error(f"❌ Prediction request failed: {exc}")

    st.divider()

    # API endpoints documentation
    st.subheader("📚 API Endpoints")
    endpoints = [
        {"method": "GET", "path": "/health", "description": "Health check"},
        {"method": "POST", "path": "/predict", "description": "Fraud risk scoring (accepts features, returns score + prediction)"},
    ]
    for ep in endpoints:
        col1, col2, col3 = st.columns([1, 3, 6])
        with col1:
            st.code(ep["method"], language="text")
        with col2:
            st.code(ep["path"], language="text")
        with col3:
            st.write(ep["description"])

    st.info("Run the API server: `uvicorn src.api:app --reload --host 0.0.0.0 --port 8000`")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**UPI Fraud Detection System**")
st.sidebar.markdown("MCA Project - Vidyavardhaka College of Engineering")
st.sidebar.markdown("Student: Pavan C (4VV25MC060)")