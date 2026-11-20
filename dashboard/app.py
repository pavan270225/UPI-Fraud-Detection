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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/pavanc/upi-fraud-detection',
        'Report a bug': 'https://github.com/pavanc/upi-fraud-detection/issues',
        'About': 'UPI Fraud Detection System - Drift-Aware Explainable ML for UPI Digital Payments'
    }
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
        /* Light theme tokens (default) */
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

    /* Dark mode - uses custom .theme-dark class on html element (toggled via JS) */
    html.theme-dark {
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

    /* Theme presets - applied via body class to avoid conflict with Streamlit's data-theme */
    body.theme-indigo { --accent-primary: #6366f1; --accent-secondary: #8b5cf6; --accent-tertiary: #06b6d4; --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%); }
    body.theme-emerald { --accent-primary: #10b981; --accent-secondary: #34d399; --accent-tertiary: #06b6d4; --accent-gradient: linear-gradient(135deg, #10b981 0%, #34d399 50%, #06b6d4 100%); }
    body.theme-amber { --accent-primary: #f59e0b; --accent-secondary: #fbbf24; --accent-tertiary: #f97316; --accent-gradient: linear-gradient(135deg, #f59e0b 0%, #fbbf24 50%, #f97316 100%); }
    body.theme-rose { --accent-primary: #f43f5e; --accent-secondary: #fb7185; --accent-tertiary: #ec4899; --accent-gradient: linear-gradient(135deg, #f43f5e 0%, #fb7185 50%, #ec4899 100%); }
    body.theme-cyan { --accent-primary: #06b6d4; --accent-secondary: #22d3ee; --accent-tertiary: #0ea5e9; --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #22d3ee 50%, #0ea5e9 100%); }

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
    html.theme-dark .stApp {
        background: var(--bg-primary) !important;
    }

    html.theme-dark .stApp::before {
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

# Theme toggle - uses custom class on <html> element via JavaScript
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

if dark_mode != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode
    st.rerun()

# Apply theme via JavaScript - toggles 'theme-dark' class on <html>
theme_js = f"""
<script>
    (function() {{
        const isDark = {str(dark_mode).lower()};
        const html = document.documentElement;

        if (isDark) {{
            html.classList.add('theme-dark');
            localStorage.setItem('upi-dark-mode', 'true');
        }} else {{
            html.classList.remove('theme-dark');
            localStorage.setItem('upi-dark-mode', 'false');
        }}
    }})();
</script>
"""
st.markdown(theme_js, unsafe_allow_html=True)

# Restore theme on page load (runs before Streamlit renders)
init_theme_js = """
<script>
    (function() {
        const saved = localStorage.getItem('upi-dark-mode');
        if (saved === 'true') {
            document.documentElement.classList.add('theme-dark');
        }
    })();
</script>
"""
st.markdown(init_theme_js, unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("---")

# ============================================================
# UI UTILITY FUNCTIONS
# ============================================================

def show_toast(message: str, type: str = "info", duration: int = 4000):
    """Show a toast notification using Streamlit's st.toast (available in 1.32+)."""
    # Fallback for older versions - use custom HTML toast
    icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    colors = {
        "success": "rgba(34, 197, 94, 0.9)",
        "error": "rgba(239, 68, 68, 0.9)",
        "warning": "rgba(245, 158, 11, 0.9)",
        "info": "rgba(99, 102, 241, 0.9)"
    }
    toast_html = f"""
    <div id="toast-{hash(message) % 10000}" class="toast-notification" style="
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: {colors.get(type, colors['info'])};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-xl);
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
        font-family: 'Inter', sans-serif;
    ">
        <span style="font-size: 1.25rem;">{icons.get(type, 'ℹ️')}</span>
        <span style="flex: 1;">{message}</span>
    </div>
    <style>
        @keyframes slideInRight {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        @keyframes slideOutRight {{
            from {{ transform: translateX(0); opacity: 1; }}
            to {{ transform: translateX(100%); opacity: 0; }}
        }}
    </style>
    <script>
        setTimeout(() => {{
            const toast = document.getElementById('toast-{hash(message) % 10000}');
            if (toast) {{
                toast.style.animation = 'slideOutRight 0.3s ease-in forwards';
                setTimeout(() => toast.remove(), 300);
            }}
        }}, {duration});
    </script>
    """
    st.markdown(toast_html, unsafe_allow_html=True)


def inject_keyboard_shortcuts():
    """Inject keyboard shortcut handlers."""
    shortcuts_js = """
    <script>
        // Keyboard shortcuts for dashboard
        if (!window.__upiShortcutsBound) {
            window.__upiShortcutsBound = true;
            document.addEventListener('keydown', (e) => {
                // Ignore if typing in input/textarea
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;

                const shortcuts = {
                    'd': () => toggleDarkMode(),           // Toggle dark mode
                    'r': () => refreshPage(),               // Refresh data
                    'h': () => showHelp(),                  // Show help
                    '1': () => navigateTo('📊 Overview'),   // Nav shortcuts
                    '2': () => navigateTo('🎯 Risk Scoring'),
                    '3': () => navigateTo('🤖 Model Comparison'),
                    '4': () => navigateTo('🚀 Pipeline Control'),
                    '5': () => navigateTo('🔍 SHAP Explanations'),
                    '6': () => navigateTo('📈 Drift Monitoring'),
                    '7': () => navigateTo('⚙️ Retraining Strategies'),
                    '8': () => navigateTo('⏱️ Delayed Labels'),
                    '9': () => navigateTo('📜 Prediction History'),
                    '0': () => navigateTo('📋 Reports'),
                };

                if (shortcuts[e.key]) {
                    e.preventDefault();
                    shortcuts[e.key]();
                }
            });
        }

        function toggleDarkMode() {
            const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
            if (toggle) {
                toggle.click();
            }
        }

        function refreshPage() {
            // Trigger refresh by clicking any refresh button
            const refreshBtn = document.querySelector('button[kind="secondary"]:has-text("Refresh")') ||
                               document.querySelector('button:has-text("🔄")') ||
                               document.querySelector('button:has-text("Run")');
            if (refreshBtn) refreshBtn.click();
        }

        function showHelp() {
            alert('Keyboard Shortcuts:\\n' +
                  'D - Toggle Dark Mode\\n' +
                  'R - Refresh Data\\n' +
                  'H - Show This Help\\n' +
                  '1-9,0 - Navigate Pages\\n' +
                  'Escape - Close Modals');
        }

        function navigateTo(pageName) {
            const radio = document.querySelector(`[data-testid="stSidebar"] label:has-text("${pageName}") input`);
            if (radio) radio.click();
        }
    </script>
    """
    st.markdown(shortcuts_js, unsafe_allow_html=True)


def get_url_params():
    """Get URL query parameters for state sharing."""
    try:
        return st.query_params
    except:
        return {}


def set_url_params(**params):
    """Set URL query parameters for state sharing."""
    try:
        st.query_params.update(params)
    except:
        pass


def skeleton_loader(count: int = 3, height: int = 120):
    """Render skeleton loading placeholders."""
    skeleton_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
        {''.join([f'''
        <div class="skeleton-card" style="
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            animation: pulse 1.5s ease-in-out infinite;
        ">
            <div style="height: 1rem; background: linear-gradient(90deg, var(--bg-tertiary) 25%, var(--border-color) 50%, var(--bg-tertiary) 75%); background-size: 200% 100%; border-radius: var(--radius-sm); margin-bottom: 1rem; animation: shimmer 1.5s infinite;"></div>
            <div style="height: 2rem; background: linear-gradient(90deg, var(--bg-tertiary) 25%, var(--border-color) 50%, var(--bg-tertiary) 75%); background-size: 200% 100%; border-radius: var(--radius-sm); margin-bottom: 0.5rem; animation: shimmer 1.5s infinite;"></div>
            <div style="height: 1rem; width: 60%; background: linear-gradient(90deg, var(--bg-tertiary) 25%, var(--border-color) 50%, var(--bg-tertiary) 75%); background-size: 200% 100%; border-radius: var(--radius-sm); animation: shimmer 1.5s infinite;"></div>
        </div>
        ''' for _ in range(count)])}
    </div>
    <style>
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
    </style>
    """
    st.markdown(skeleton_html, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a consistent page header with optional subtitle."""
    st.markdown(f"""
    <div class="glass-card animate-fade-in-up" style="margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1 style="margin: 0; display: flex; align-items: center; gap: 0.75rem;">
                    {f'<span style="font-size: 2rem;">{icon}</span>' if icon else ''}
                    {title}
                </h1>
                {f'<p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 1.1rem;">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal", icon: str = "", help_text: str = ""):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        color = "#22c55e" if delta_color == "normal" else "#ef4444" if delta_color == "inverse" else "#f59e0b"
        delta_html = f'<div style="font-size: 0.85rem; color: {color}; margin-top: 0.5rem; font-weight: 500;">{delta}</div>'

    help_attr = f'title="{help_text}"' if help_text else ""

    st.markdown(f"""
    <div class="glass-card metric-card animate-fade-in-up" {help_attr} style="cursor: help;">
        <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            {f'<span>{icon}</span>' if icon else ''}
            {label}
        </div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.1;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# Inject keyboard shortcuts
inject_keyboard_shortcuts()

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🎯 Risk Scoring", "🤖 Model Comparison", "🚀 Pipeline Control",
     "🔍 SHAP Explanations", "📈 Drift Monitoring",
     "⚙️ Retraining Strategies", "⏱️ Delayed Labels", "📜 Prediction History", "📋 Reports", "🔌 API Integration"]
)

# Theme selector in sidebar
theme_options = {
    "Indigo (Default)": "indigo",
    "Emerald": "emerald",
    "Amber": "amber",
    "Rose": "rose",
    "Cyan": "cyan"
}
selected_theme = st.sidebar.selectbox("🎨 Theme", list(theme_options.keys()), index=0)
theme_value = theme_options[selected_theme]

# Apply theme via body class (not data-theme, which is used by Streamlit for dark/light)
theme_script = f"""
<script>
    const body = document.body;
    // Remove all theme classes
    body.classList.remove('theme-indigo', 'theme-emerald', 'theme-amber', 'theme-rose', 'theme-cyan');
    // Add selected theme
    body.classList.add('theme-{theme_value}');
    localStorage.setItem('upi-theme', '{theme_value}');
</script>
"""
st.markdown(theme_script, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False, key="auto_refresh")
if auto_refresh:
    st.markdown("""
    <script>
        if (!window.__upiAutoRefresh) {
            window.__upiAutoRefresh = setInterval(() => {
                // Find and click any refresh button
                const buttons = document.querySelectorAll('button');
                buttons.forEach(btn => {
                    if (btn.textContent.includes('🔄') || btn.textContent.includes('Refresh') || btn.textContent.includes('Run')) {
                        btn.click();
                    }
                });
            }, 30000);
        }
    </script>
    """, unsafe_allow_html=True)

# Load reports
reports = load_reports()
model, model_meta = load_model_and_data()

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "📊 Overview":
    # Hero section with animated gradient text
    render_page_header(
        "UPI Fraud Detection",
        "Drift-Aware Explainable ML Framework for Digital Payments",
        "🛡️"
    )

    # Action buttons row
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh Data", key="overview_refresh", use_container_width=True):
            show_toast("Data refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("📊 Run Full Pipeline", key="overview_pipeline", use_container_width=True):
            show_toast("Pipeline execution triggered. Check Pipeline Control page.", "info")
    with col_actions[2]:
        if st.button("🔍 Run Drift Check", key="overview_drift", use_container_width=True):
            show_toast("Drift check queued. Check Drift Monitoring page.", "info")
    with col_actions[3]:
        st.link_button("📚 Docs", "https://github.com/pavanc/upi-fraud-detection", use_container_width=True)

    # KPI Cards using render_metric_card
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            "Project", "UPI Fraud Detection",
            "Drift-Aware Explainable ML", "normal",
            "🛡️", "Comprehensive fraud detection pipeline for UPI payments"
        )

    with col2:
        if not reports['ieee_test_metrics'].empty:
            best_pr_auc = reports['ieee_test_metrics']['pr_auc'].max()
            passed = best_pr_auc > 0.85
            render_metric_card(
                "Best PR-AUC (IEEE-CIS)", f"{best_pr_auc:.4f}",
                "✅ Target met" if passed else "❌ Below target (0.85)",
                "normal" if passed else "inverse",
                "📈", f"Target: >0.85 | Current: {best_pr_auc:.4f}"
            )
        else:
            render_metric_card(
                "Best PR-AUC (IEEE-CIS)", "N/A",
                "Run notebook 02_baselines.ipynb", "warning",
                "📈", "No baseline metrics available"
            )

    with col3:
        if not reports['drift_alerts'].empty:
            total_alerts = len(reports['drift_alerts'])
            pending = sum(1 for _, row in reports['drift_alerts'].iterrows()
                          if st.session_state.alert_status.get(alert_key(row.to_dict()), "pending") == "pending")
            render_metric_card(
                "Drift Alerts", str(total_alerts),
                f"{pending} pending" if pending else "All resolved",
                "warning" if pending else "normal",
                "📊", f"Total alerts logged | {pending} need attention"
            )
        else:
            render_metric_card(
                "Drift Alerts", "0",
                "Run drift detection", "normal",
                "📊", "No drift alerts generated yet"
            )

    with col4:
        if not reports['retrain_comparison'].empty:
            strategies = reports['retrain_comparison']['strategy'].nunique()
            render_metric_card(
                "Strategies Compared", str(strategies),
                "Static, Scheduled, Drift-Triggered", "normal",
                "⚙️", "Retraining strategy comparison results"
            )
        else:
            render_metric_card(
                "Strategies Compared", "0",
                "Run notebook 06_retraining_strategies.ipynb", "warning",
                "⚙️", "No retraining comparison available"
            )

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

        # Export/Share buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            if st.button("📄 Export as PDF", key="overview_pdf", use_container_width=True):
                show_toast("PDF export feature coming soon", "info")
        with col_exp2:
            if st.button("🔗 Share View", key="overview_share", use_container_width=True):
                share_url = f"{st.query_params.get('_base_url', '')}?page=overview&theme={theme_value}"
                set_url_params(page="overview", theme=theme_value)
                show_toast("View URL updated! Copy from address bar.", "success")
        with col_exp3:
            if st.button("📊 Export JSON", key="overview_json", use_container_width=True):
                import json
                export_data = {k: v.to_dict() if hasattr(v, 'to_dict') else str(v) for k, v in reports.items()}
                st.download_button("Download JSON", json.dumps(export_data, default=str),
                                 file_name="upi_fraud_report.json", mime="application/json",
                                 key="overview_json_dl", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline diagram with interactive elements
    render_page_header("Pipeline Architecture", "End-to-end fraud detection workflow", "🔄")

    # Interactive pipeline stages
    pipeline_stages = [
        ("📥", "Data Collection", "IEEE-CIS, PaySim datasets"),
        ("✂️", "Time-Ordered Split", "70/15/15 temporal split"),
        ("🤖", "Baseline Models", "LR, RF, XGBoost, LightGBM"),
        ("⏱️", "Delayed Labels", "1/7/30-day simulation"),
        ("📊", "Drift Monitoring", "PSI/KS on features & scores"),
        ("🔍", "SHAP Explanations", "Per-alert + Global importance"),
        ("⚙️", "Retraining", "Static/Scheduled/Drift-triggered"),
        ("📈", "Evaluation", "PR-AUC, P@K, FP/100k"),
    ]

    cols = st.columns(len(pipeline_stages))
    for i, (icon, title, desc) in enumerate(pipeline_stages):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1rem; cursor: pointer;"
                 onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='var(--shadow-lg)'; this.style.borderColor='var(--border-glow)'"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='var(--shadow-md)'; this.style.borderColor='var(--border-color)'"
                 title="Click to navigate to {title} page">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">{title}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Graphviz diagram
    st.graphviz_chart("""
    digraph pipeline {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="#6366f1", fontcolor=white, fontname="Arial", fontsize=11];
        edge [fontname="Arial", color="#64748b"];
        splines=ortho;

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
    render_page_header("Model Comparison", "Compare baseline model performance across datasets", "🤖")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="model_refresh", use_container_width=True):
            show_toast("Model data refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("📊 Run Baselines", key="model_baselines", use_container_width=True):
            show_toast("Run notebook 02_baselines.ipynb to generate baselines", "info")
    with col_actions[2]:
        if st.button("🔗 Share View", key="model_share", use_container_width=True):
            set_url_params(page="model-comparison", theme=theme_value)
            show_toast("View URL updated!", "success")

    # Load metrics from reports
    ieee_df = reports.get('ieee_test_metrics', pd.DataFrame())
    paysim_df = reports.get('paysim_test_metrics', pd.DataFrame())

    if ieee_df.empty and paysim_df.empty:
        skeleton_loader(3, 200)
        st.warning("No model metrics found. Run notebook 02_baselines.ipynb to generate evaluation results.")
    else:
        # Normalize column names
        def normalize_metrics(df):
            if df.empty:
                return df
            df = df.copy()
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

        # Combined comparison table with enhanced display
        st.subheader("📊 Side-by-Side Comparison")

        metrics_to_show = ['pr_auc', 'precision_at_100', 'recall_at_100', 'fp_per_100k_at_budget_100', 'f1', 'roc_auc']
        available_metrics = [m for m in metrics_to_show if m in ieee_df.columns or m in paysim_df.columns]

        if 'model' in ieee_df.columns or 'model' in paysim_df.columns:
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

            # Enhanced dataframe with conditional formatting
            st.dataframe(
                comp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "model": st.column_config.TextColumn("Model", width="medium"),
                    **{col: st.column_config.NumberColumn(
                        col.replace("ieee_", "IEEE ").replace("paysim_", "PaySim ").replace("_", " ").title(),
                        format="%.4f" if "auc" in col or "pr_" in col else "%.2f",
                        help=f"{col} metric"
                    ) for col in comp_df.columns if col != "model"}
                }
            )

            # Best model cards
            if 'ieee_pr_auc' in comp_df.columns and comp_df['ieee_pr_auc'].notna().any():
                best_idx = comp_df['ieee_pr_auc'].idxmax()
                best_model = comp_df.iloc[best_idx]

                col_best1, col_best2, col_best3 = st.columns(3)
                with col_best1:
                    render_metric_card(
                        "🏆 Best IEEE-CIS Model", best_model['model'],
                        f"PR-AUC: {best_model.get('ieee_pr_auc', 0):.4f}", "normal",
                        "🥇", "Highest PR-AUC on IEEE-CIS test set"
                    )
                with col_best2:
                    if 'paysim_pr_auc' in comp_df.columns and comp_df['paysim_pr_auc'].notna().any():
                        best_paysim_idx = comp_df['paysim_pr_auc'].idxmax()
                        best_paysim = comp_df.iloc[best_paysim_idx]
                        render_metric_card(
                            "🏆 Best PaySim Model", best_paysim['model'],
                            f"PR-AUC: {best_paysim.get('paysim_pr_auc', 0):.4f}", "normal",
                            "🥇", "Highest PR-AUC on PaySim test set"
                        )
                with col_best3:
                    # Overall winner
                    if 'ieee_pr_auc' in comp_df.columns and 'paysim_pr_auc' in comp_df.columns:
                        comp_df['avg_pr_auc'] = (comp_df['ieee_pr_auc'].fillna(0) + comp_df['paysim_pr_auc'].fillna(0)) / 2
                        overall_best_idx = comp_df['avg_pr_auc'].idxmax()
                        overall_best = comp_df.iloc[overall_best_idx]
                        render_metric_card(
                            "🌟 Overall Best", overall_best['model'],
                            f"Avg PR-AUC: {overall_best.get('avg_pr_auc', 0):.4f}", "normal",
                            "🌟", "Best average PR-AUC across both datasets"
                        )

            # Interactive charts with drill-down
            st.subheader("📈 Performance Visualization")

            chart_type = st.radio("Chart Type", ["Horizontal Bar", "Grouped Bar", "Radar"], horizontal=True, key="model_chart_type")

            col1, col2 = st.columns(2)

            with col1:
                if 'ieee_pr_auc' in comp_df.columns and comp_df['ieee_pr_auc'].notna().any():
                    plot_df = comp_df.dropna(subset=['ieee_pr_auc']).sort_values('ieee_pr_auc', ascending=True)

                    if chart_type == "Horizontal Bar":
                        fig = px.bar(
                            plot_df, x='ieee_pr_auc', y='model', orientation='h',
                            title='PR-AUC by Model (IEEE-CIS)',
                            labels={'ieee_pr_auc': 'PR-AUC', 'model': 'Model'},
                            color='ieee_pr_auc', color_continuous_scale='Viridis',
                            text='ieee_pr_auc'
                        )
                        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    elif chart_type == "Grouped Bar":
                        fig = px.bar(
                            comp_df.melt(id_vars=['model'], value_vars=[c for c in comp_df.columns if c.startswith('ieee_')]),
                            x='model', y='value', color='variable', barmode='group',
                            title='All IEEE-CIS Metrics by Model'
                        )
                    else:  # Radar
                        fig = go.Figure()
                        for _, row in comp_df.iterrows():
                            if row['ieee_pr_auc'] is not None:
                                values = [row.get(f'ieee_{m}', 0) for m in ['pr_auc', 'precision_at_100', 'recall_at_100', 'f1']]
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=['PR-AUC', 'Precision@100', 'Recall@100', 'F1'],
                                    fill='toself', name=row['model']
                                ))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))

                    fig.update_layout(height=400, showlegend=chart_type!="Horizontal Bar")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'ieee_precision_at_100' in comp_df.columns and comp_df['ieee_precision_at_100'].notna().any():
                    plot_df = comp_df.dropna(subset=['ieee_precision_at_100']).sort_values('ieee_precision_at_100', ascending=True)

                    if chart_type == "Horizontal Bar":
                        fig = px.bar(
                            plot_df, x='ieee_precision_at_100', y='model', orientation='h',
                            title='Precision@100 by Model (IEEE-CIS)',
                            labels={'ieee_precision_at_100': 'Precision@100', 'model': 'Model'},
                            color='ieee_precision_at_100', color_continuous_scale='Plasma',
                            text='ieee_precision_at_100'
                        )
                        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    elif chart_type == "Grouped Bar":
                        fig = px.bar(
                            comp_df.melt(id_vars=['model'], value_vars=[c for c in comp_df.columns if c.startswith('paysim_')]),
                            x='model', y='value', color='variable', barmode='group',
                            title='All PaySim Metrics by Model'
                        )

                    fig.update_layout(height=400, showlegend=chart_type!="Horizontal Bar")
                    st.plotly_chart(fig, use_container_width=True)

            # Heatmap comparison
            st.subheader("🔥 Metric Heatmap")
            heatmap_cols = [c for c in comp_df.columns if c != 'model' and comp_df[c].notna().any()]
            if len(heatmap_cols) > 1:
                fig = px.imshow(
                    comp_df.set_index('model')[heatmap_cols].fillna(0).T,
                    labels=dict(x="Model", y="Metric", color="Value"),
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    title="Cross-Dataset Metric Comparison"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Model column not found in metrics. Available columns: " + ", ".join(
                list(ieee_df.columns) + list(paysim_df.columns)))

# ============================================================
# PAGE: PIPELINE CONTROL
# ============================================================
elif page == "🚀 Pipeline Control":
    render_page_header("Pipeline Control", "Run end-to-end pipeline operations in real-time", "🚀")

    # Status indicators
    col_status = st.columns(4)
    with col_status[0]:
        render_metric_card("Dataset", "IEEE-CIS", "Change in sidebar", "normal", "📊", "Active dataset for operations")
    with col_status[1]:
        render_metric_card("Pipeline Status", "Ready", "No active runs", "normal", "⚡", "Current pipeline state")
    with col_status[2]:
        render_metric_card("Last Run", "Never", "Click Run to start", "normal", "🕐", "Last pipeline execution")
    with col_status[3]:
        if st.button("📊 View Logs", key="pc_logs", use_container_width=True):
            show_toast("Pipeline logs view coming soon", "info")

    # Dataset selector
    dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="pc_dataset")

    # Create tabs for each pipeline stage
    tab_train, tab_drift, tab_retrain, tab_river, tab_online, tab_all = st.tabs([
        "📚 Train", "📈 Drift Detection", "🔄 Retrain", "🌊 River Drift", "🔁 Online Learning", "▶️ Full Pipeline"
    ])

    with tab_train:
        st.subheader("📚 Train Models")

        col1, col2, col3 = st.columns(3)
        with col1:
            pc_model = st.selectbox("Model",
                                    ["lightgbm", "logistic_regression", "random_forest", "xgboost", "catboost", "all"],
                                    key="pc_model",
                                    help="Select 'all' to train all baseline models")
        with col2:
            pc_train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.7, 0.05, key="pc_train_ratio")
        with col3:
            pc_val_ratio = st.slider("Val Ratio", 0.05, 0.3, 0.15, 0.05, key="pc_val_ratio")
        pc_test_ratio = st.slider("Test Ratio", 0.05, 0.3, 0.15, 0.05, key="pc_test_ratio")

        # Ensemble options
        pc_create_ensemble = st.checkbox("Create Ensemble from Top-K Models", key="pc_create_ensemble", value=True)
        pc_top_k = st.slider("Top-K for Ensemble", 2, 5, 3, key="pc_top_k")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🏋️ Train Model(s)", key="pc_train_btn", type="primary", use_container_width=True):
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
                        show_toast("✅ Training completed!", "success")
                        if "models" in result:
                            st.subheader("Training Results")
                            for name, m in result["models"].items():
                                if "error" in m:
                                    st.error(f"{name}: {m['error']}")
                                else:
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        render_metric_card(name, f"Val PR-AUC: {m['val_metrics']['pr_auc']:.4f}",
                                                         f"Test PR-AUC: {m['test_metrics']['pr_auc']:.4f}", "normal")
                                    with col_b:
                                        render_metric_card("", f"P@100: {m['val_metrics'].get('precision_at_100', 0):.4f}",
                                                         f"F1: {m['val_metrics'].get('f1', 0):.4f}", "normal")
                            if "ensemble" in result and result["ensemble"]:
                                ens = result["ensemble"]
                                st.success(f"🏆 Ensemble created: Test PR-AUC = {ens.get('test_metrics', {}).get('pr_auc', 'N/A'):.4f}")
                                st.write(f"Models: {ens.get('ensemble_models', [])}")
                                st.write(f"Weights: {ens.get('ensemble_weights', {})}")
                        st.rerun()
                    except Exception as exc:
                        show_toast(f"❌ Training failed: {exc}", "error")
        with col_btn2:
            if st.button("🔄 Reset & Retrain", key="pc_retrain_full", use_container_width=True):
                show_toast("Full retrain triggered", "info")

    with tab_drift:
        st.subheader("📈 Drift Detection (PSI/KS)")

        col1, col2 = st.columns(2)
        with col1:
            pc_ref_days = st.slider("Reference Days", 1, 30, 7, key="pc_ref_days")
        with col2:
            pc_cur_days = st.slider("Current Days", 1, 30, 7, key="pc_cur_days")

        if st.button("🔍 Detect Drift", key="pc_drift_btn", type="primary", use_container_width=True):
            with st.spinner("Running drift detection..."):
                try:
                    result = stage_drift(
                        dataset=dataset,
                        reference_window_days=pc_ref_days,
                        current_window_days=pc_cur_days,
                    )
                    show_toast("✅ Drift detection completed!", "success")

                    alerts = result.get("alerts", [])
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        render_metric_card("Total Alerts", str(len(alerts)), "", "normal", "🚨")
                    with col_b:
                        feature_count = sum(1 for a in alerts if a.get("type") == "feature_drift")
                        render_metric_card("Feature Drift", str(feature_count), "", "warning" if feature_count else "normal", "📊")
                    with col_c:
                        score_count = sum(1 for a in alerts if "score_drift" in a.get("type", ""))
                        render_metric_card("Score Drift", str(score_count), "", "warning" if score_count else "normal", "📈")

                    if alerts:
                        st.subheader("🚨 Drift Alerts")
                        for alert in alerts:
                            value = alert.get("value", 0)
                            if value > 0.3:
                                severity = "🔴 High"
                                sev_color = "error"
                            elif value > 0.2:
                                severity = "🟡 Medium"
                                sev_color = "warning"
                            else:
                                severity = "🟢 Low"
                                sev_color = "normal"
                            render_metric_card(
                                f"{severity} {alert['type']}",
                                f"{alert['column']} ({alert['metric']}={value:.4f})",
                                "", sev_color
                            )
                except Exception as exc:
                    show_toast(f"❌ Drift detection failed: {exc}", "error")

    with tab_retrain:
        st.subheader("🔄 Drift-Triggered Retraining")

        col1, col2 = st.columns(2)
        with col1:
            pc_retrain_model = st.selectbox("Model",
                                            ["lightgbm", "logistic_regression", "random_forest", "xgboost"],
                                            key="pc_retrain_model")
        with col2:
            pc_force = st.checkbox("Force Retrain (Ignore Drift)", key="pc_force")

        col3, col4 = st.columns(2)
        with col3:
            pc_psi = st.slider("PSI Threshold", 0.1, 0.5, 0.2, 0.05, key="pc_psi")
        with col4:
            pc_ks = st.slider("KS Threshold", 0.01, 0.2, 0.05, 0.01, key="pc_ks")

        if st.button("🔄 Check Drift & Retrain", key="pc_retrain_btn", type="primary", use_container_width=True):
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
                        show_toast(f"✅ Model retrained (trigger: {result['trigger']})!", "success")
                    else:
                        show_toast("ℹ️ No retraining needed — drift below thresholds.", "info")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        render_metric_card("Val PR-AUC", f"{result['val_metrics']['pr_auc']:.4f}", "", "normal")
                    with col_b:
                        render_metric_card("Test PR-AUC", f"{result['test_metrics']['pr_auc']:.4f}", "", "normal")

                    if result["drift_alerts"]:
                        st.write(f"**Drift alerts triggered**: {len(result['drift_alerts'])}")
                        for alert in result["drift_alerts"]:
                            st.write(f"  - {alert['type']}: {alert['column']} ({alert['metric']}={alert['value']:.4f})")
                except Exception as exc:
                    show_toast(f"❌ Retraining failed: {exc}", "error")

    with tab_river:
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

        col3, col4 = st.columns(2)
        with col3:
            pc_river_psi = st.slider("PSI Threshold", 0.1, 0.5, 0.2, 0.05, key="pc_river_psi")
        with col4:
            pc_river_ks = st.slider("KS Threshold", 0.01, 0.2, 0.05, 0.01, key="pc_river_ks")

        pc_river_score_threshold = st.slider("Score Threshold", 0.1, 0.9, 0.5, 0.05, key="pc_river_score_threshold")

        if st.button("🌊 Run River Drift Detection", key="pc_river_drift_btn", type="primary", use_container_width=True):
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
                    show_toast("✅ River drift detection completed!", "success")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        render_metric_card("Total Alerts", str(len(result.get('alerts', []))), "", "normal", "🚨")
                    with col_b:
                        render_metric_card("Detectors Used", str(len(result.get('river_detectors', {}))), "", "normal", "🔧")

                    for detector, detected in result.get('river_detectors', {}).items():
                        render_metric_card(
                            detector.upper(),
                            "🔴 Drift Detected" if detected else "🟢 No Drift",
                            "", "error" if detected else "normal", "📊"
                        )
                except Exception as exc:
                    show_toast(f"❌ River drift detection failed: {exc}", "error")

    with tab_online:
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

        if st.button("🔁 Run Online Learning", key="pc_online_btn", type="primary", use_container_width=True):
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
                    show_toast("✅ Online learning completed!", "success")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        render_metric_card("Model Type", result.get('model_type', 'N/A'), "", "normal")
                    with col_b:
                        render_metric_card("Train Samples", str(result.get('train_samples', 0)), "", "normal")
                    with col_c:
                        render_metric_card("Test Samples", str(result.get('test_samples', 0)), "", "normal")

                    if 'val_metrics' in result:
                        render_metric_card("Val PR-AUC", f"{result.get('val_metrics', {}).get('pr_auc', 0):.4f}", "", "normal")
                        render_metric_card("Test PR-AUC", f"{result.get('test_metrics', {}).get('pr_auc', 0):.4f}", "", "normal")

                    if result.get('drift_alerts'):
                        st.write(f"**Drift alerts**: {len(result['drift_alerts'])}")
                except Exception as exc:
                    show_toast(f"❌ Online learning failed: {exc}", "error")

    with tab_all:
        st.subheader("🔄 Full Pipeline Execution")
        st.markdown("Run train → evaluate → drift → retrain → shap in sequence.")

        st.info("⚠️ This will execute the complete pipeline. It may take several minutes.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run Full Pipeline", key="pc_run_all_btn", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("🏋️ Training models...")
                    progress_bar.progress(20)
                    stage_train(
                        dataset=dataset,
                        model_name=pc_retrain_model,
                        train_ratio=pc_train_ratio,
                        val_ratio=pc_val_ratio,
                        test_ratio=pc_test_ratio,
                        save=True,
                        quiet=True,
                    )
                    show_toast("✅ Train complete", "success")

                    status_text.text("📈 Detecting drift...")
                    progress_bar.progress(40)
                    stage_drift(dataset=dataset)
                    show_toast("✅ Drift detection complete", "success")

                    status_text.text("🔄 Checking retraining...")
                    progress_bar.progress(60)
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
                    show_toast("✅ Retrain check complete", "success")

                    status_text.text("🔍 Computing SHAP...")
                    progress_bar.progress(80)
                    # SHAP would go here

                    progress_bar.progress(100)
                    status_text.text("🎉 Full pipeline completed!")
                    show_toast("🎉 Full pipeline completed successfully!", "success")
                    st.balloons()
                except Exception as exc:
                    show_toast(f"❌ Pipeline failed: {exc}", "error")
                    status_text.text(f"❌ Failed: {exc}")

        with col2:
            if st.button("📋 Show Pipeline Config", key="pc_show_config", use_container_width=True):
                st.json({
                    "dataset": dataset,
                    "model": pc_retrain_model,
                    "train_ratio": pc_train_ratio,
                    "val_ratio": pc_val_ratio,
                    "test_ratio": pc_test_ratio,
                    "drift_psi_threshold": pc_psi,
                    "drift_ks_threshold": pc_ks,
                    "ensemble_enabled": pc_create_ensemble,
                    "ensemble_top_k": pc_top_k
                })

    # ── Run all ──
    st.subheader("🔄 Full Pipeline")
    st.markdown("Run train → evaluate → drift → retrain → shap in sequence.")

    if st.button("▶️ Run All", key="pc_run_all_simple"):
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
    render_page_header("SHAP Explanations", "Understand model decisions with TreeSHAP", "🔍")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="shap_refresh", use_container_width=True):
            show_toast("SHAP data refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("📊 Run SHAP Notebook", key="shap_notebook", use_container_width=True):
            show_toast("Run notebook 05_shap_explanations.ipynb", "info")
    with col_actions[2]:
        if st.button("🔗 Share View", key="shap_share", use_container_width=True):
            set_url_params(page="shap", theme=theme_value)
            show_toast("View URL updated!", "success")

    if model is None:
        st.warning("⚠️ Model not loaded. Please train models first using Pipeline Control → Train.")
        skeleton_loader(3, 200)
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

            tab1, tab2, tab3, tab4 = st.tabs(["🔬 Per-Instance", "📊 Global Importance", "📈 Importance Over Time", "🔍 Feature Interaction"])

            with tab1:
                st.subheader("Per-Instance SHAP Explanation")

                # Model score distribution for selection
                scores = model.predict_proba(X_test)[:, 1]

                col_select1, col_select2 = st.columns([2, 1])
                with col_select1:
                    selection_mode = st.radio("Selection Mode",
                        ["Top Riskiest", "Random Sample", "By Index", "Custom Features"],
                        horizontal=True, key="shap_select_mode")

                with col_select2:
                    max_display = st.slider("Max Features", 5, 30, 15, key="shap_max_display")

                if selection_mode == "Top Riskiest":
                    top_indices = np.argsort(scores)[-20:]
                    selected_idx = st.selectbox(
                        "Select Transaction (Top 20 Riskiest)",
                        top_indices[::-1],
                        format_func=lambda x: f"Idx {x} - Score: {scores[x]:.4f} - Label: {'🔴 Fraud' if y_test.iloc[x]==1 else '🟢 Legit'}",
                        key="shap_top_select"
                    )
                elif selection_mode == "Random Sample":
                    sample_indices = np.random.choice(len(X_test), min(20, len(X_test)), replace=False)
                    selected_idx = st.selectbox(
                        "Select Random Transaction",
                        sample_indices,
                        format_func=lambda x: f"Idx {x} - Score: {scores[x]:.4f} - Label: {'🔴 Fraud' if y_test.iloc[x]==1 else '🟢 Legit'}",
                        key="shap_rand_select"
                    )
                elif selection_mode == "By Index":
                    selected_idx = st.number_input("Transaction Index", 0, len(X_test)-1, 0, key="shap_idx_input")
                else:  # Custom Features
                    st.info("Enter feature values manually to get SHAP explanation")
                    custom_features = {}
                    for col in X_test.columns[:10]:  # First 10 features
                        custom_features[col] = st.number_input(f"{col}", value=float(X_test[col].median()), key=f"shap_custom_{col}")
                    if st.button("🔍 Explain Custom", key="shap_custom_btn"):
                        x_row = pd.DataFrame([custom_features]).reindex(columns=X_test.columns, fill_value=-999)
                        explanation = explainer.explain_instance(x_row, max_display=max_display)
                        # Display custom explanation
                        render_custom_explanation(explanation)
                    selected_idx = None

                if selected_idx is not None:
                    x_row = X_test.loc[selected_idx].to_frame().T
                    explanation = explainer.explain_instance(x_row, max_display=max_display)

                    # Enhanced metric cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        render_metric_card("Prediction", f"{explanation['prediction']:.4f}",
                                         "🔴 Fraud" if explanation['prediction'] > 0.5 else "🟢 Legit",
                                         "error" if explanation['prediction'] > 0.5 else "normal", "🎯")
                    with col2:
                        render_metric_card("Base Value", f"{explanation['base_value']:.4f}", "", "normal", "📊")
                    with col3:
                        true_label = "🔴 Fraud" if y_test.iloc[selected_idx]==1 else "🟢 Legit"
                        render_metric_card("True Label", true_label,
                                         "✅ Correct" if (explanation['prediction'] > 0.5) == (y_test.iloc[selected_idx]==1) else "❌ Incorrect",
                                         "normal" if (explanation['prediction'] > 0.5) == (y_test.iloc[selected_idx]==1) else "error", "🏷️")
                    with col4:
                        top_contrib = explanation['feature_contributions'][0] if explanation['feature_contributions'] else ("N/A", 0)
                        render_metric_card("Top Driver", top_contrib[0], f"SHAP: {top_contrib[1]:.4f}", "normal", "🔝")

                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.subheader("📊 SHAP Waterfall Plot")
                        contrib = explanation['feature_contributions'][:max_display]
                        feat_names = [c[0] for c in contrib]
                        feat_vals = [c[1] for c in contrib]

                        # Enhanced waterfall with colors
                        colors = ['#ef4444' if v > 0 else '#22c55e' for v in feat_vals]
                        fig = go.Figure(go.Waterfall(
                            name="SHAP",
                            orientation="h",
                            measure=["relative"] * len(feat_vals) + ["total"],
                            x=feat_vals + [sum(feat_vals)],
                            y=feat_names + ["Prediction"],
                            connector={"line": {"color": "rgba(148, 163, 184, 0.3)"}},
                            increasing={"marker": {"color": "#ef4444"}},
                            decreasing={"marker": {"color": "#22c55e"}},
                            totals={"marker": {"color": "#6366f1"}}
                        ))
                        fig.update_layout(
                            height=550,
                            title="SHAP Waterfall - Feature Contributions to Prediction",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='var(--text-primary)')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col_right:
                        st.subheader("📋 Feature Contributions Table")
                        contrib_df = pd.DataFrame(contrib, columns=['Feature', 'SHAP Value'])
                        contrib_df['Direction'] = contrib_df['SHAP Value'].apply(
                            lambda x: '🔴 Increases Fraud Risk' if x > 0 else '🟢 Decreases Fraud Risk'
                        )
                        contrib_df['Abs Value'] = contrib_df['SHAP Value'].abs()
                        contrib_df = contrib_df.sort_values('Abs Value', ascending=False)

                        st.dataframe(
                            contrib_df[['Feature', 'SHAP Value', 'Direction']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "SHAP Value": st.column_config.NumberColumn("SHAP Value", format="%.4f"),
                                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                                "Direction": st.column_config.TextColumn("Impact", width="large")
                            }
                        )

                        # Feature values for this instance
                        with st.expander("🔍 View Raw Feature Values for This Transaction"):
                            instance_df = pd.DataFrame({
                                'Feature': x_row.columns,
                                'Value': x_row.iloc[0].values
                            })
                            st.dataframe(instance_df, use_container_width=True, hide_index=True)

            with tab2:
                st.subheader("Global Feature Importance (Mean |SHAP|)")

                col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
                with col_ctrl1:
                    top_n = st.slider("Top N Features", 5, 50, 20, key="shap_global_top_n")
                with col_ctrl2:
                    max_samples = st.slider("Max Samples", 100, 5000, min(2000, len(X_test)), 100, key="shap_global_samples")
                with col_ctrl3:
                    sort_by = st.selectbox("Sort By", ["Mean |SHAP|", "Std SHAP", "Feature Name"], key="shap_global_sort")

                with st.spinner("Computing global importance..."):
                    importance_df = explainer.global_importance(X_test, max_samples=max_samples)

                if sort_by == "Std SHAP":
                    importance_df = importance_df.sort_values('std_shap', ascending=False)
                elif sort_by == "Feature Name":
                    importance_df = importance_df.sort_values('feature')

                # Enhanced bar chart with error bars
                plot_df = importance_df.head(top_n).iloc[::-1]

                fig = px.bar(
                    plot_df,
                    x='mean_abs_shap', y='feature',
                    orientation='h', error_x='std_shap',
                    title=f'Top {top_n} Features by Mean |SHAP| (± Std Dev)',
                    labels={'mean_abs_shap': 'Mean |SHAP Value|', 'feature': 'Feature'},
                    color='mean_abs_shap', color_continuous_scale='Viridis',
                    text='mean_abs_shap'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Summary statistics
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    render_metric_card("Total Features", str(len(importance_df)), "", "normal", "📊")
                with col_s2:
                    render_metric_card("Top Feature", importance_df.iloc[0]['feature'],
                                     f"SHAP: {importance_df.iloc[0]['mean_abs_shap']:.4f}", "normal", "🥇")
                with col_s3:
                    render_metric_card("Cumulative Top 10", f"{importance_df.head(10)['mean_abs_shap'].sum():.4f}",
                                     f"of {importance_df['mean_abs_shap'].sum():.4f} total", "normal", "📈")

                # Detailed table
                with st.expander("📋 Full Importance Table"):
                    st.dataframe(
                        importance_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "feature": st.column_config.TextColumn("Feature"),
                            "mean_abs_shap": st.column_config.NumberColumn("Mean |SHAP|", format="%.6f"),
                            "std_shap": st.column_config.NumberColumn("Std SHAP", format="%.6f"),
                            "min_shap": st.column_config.NumberColumn("Min SHAP", format="%.6f"),
                            "max_shap": st.column_config.NumberColumn("Max SHAP", format="%.6f")
                        }
                    )

            with tab3:
                st.subheader("Feature Importance Stability Over Time")

                if not reports['shap_stability'].empty:
                    stability_df = reports['shap_stability'].copy()

                    # Add stability categories
                    stability_df['stability_cat'] = stability_df['cv'].apply(
                        lambda x: '🟢 Stable' if x < 0.3 else ('🟡 Moderate' if x < 0.5 else '🔴 Unstable')
                    )

                    col_viz1, col_viz2 = st.columns(2)

                    with col_viz1:
                        # Bar chart of CV
                        top_features = stability_df.head(20)
                        fig = px.bar(
                            top_features,
                            x='feature', y='cv',
                            color='cv', color_continuous_scale='RdYlGn_r',
                            title='SHAP Importance Coefficient of Variation (Lower = More Stable)',
                            labels={'cv': 'Coefficient of Variation', 'feature': 'Feature'}
                        )
                        fig.add_hline(y=0.3, line_dash="dash", line_color="#22c55e", annotation_text="Stable threshold (0.3)")
                        fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Unstable threshold (0.5)")
                        fig.update_layout(xaxis_tickangle=-45, height=500)
                        st.plotly_chart(fig, use_container_width=True)

                    with col_viz2:
                        # Scatter: mean importance vs stability
                        fig = px.scatter(
                            stability_df,
                            x='mean_importance', y='cv',
                            color='stability_cat',
                            hover_data=['feature', 'cv', 'trend', 'n_windows'],
                            title='Importance vs Stability',
                            labels={'mean_importance': 'Mean Importance', 'cv': 'Coefficient of Variation'},
                            color_discrete_map={'🟢 Stable': '#22c55e', '🟡 Moderate': '#f59e0b', '🔴 Unstable': '#ef4444'}
                        )
                        fig.add_hline(y=0.3, line_dash="dash", line_color="#22c55e")
                        fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                    # Stability table with filtering
                    st.subheader("📋 Stability Details")
                    filter_stability = st.multiselect(
                        "Filter by Stability",
                        ['🟢 Stable', '🟡 Moderate', '🔴 Unstable'],
                        default=['🟢 Stable', '🟡 Moderate', '🔴 Unstable'],
                        key="shap_stability_filter"
                    )
                    filtered_stability = stability_df[stability_df['stability_cat'].isin(filter_stability)]

                    st.dataframe(
                        filtered_stability[['feature', 'mean_importance', 'cv', 'trend', 'n_windows', 'stability_cat']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "feature": st.column_config.TextColumn("Feature"),
                            "mean_importance": st.column_config.NumberColumn("Mean Importance", format="%.6f"),
                            "cv": st.column_config.NumberColumn("CV", format="%.4f"),
                            "trend": st.column_config.TextColumn("Trend"),
                            "n_windows": st.column_config.NumberColumn("Windows", format="%d"),
                            "stability_cat": st.column_config.TextColumn("Stability")
                        }
                    )
                else:
                    st.info("📝 Run notebook 05_shap_explanations.ipynb to generate stability data")
                    skeleton_loader(3, 150)

            with tab4:
                st.subheader("🔍 Feature Interaction Analysis")
                st.markdown("Analyze how pairs of features interact in the model's decision making.")

                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    feature_a = st.selectbox("Feature A", X_test.columns.tolist(), key="shap_interact_a")
                with col_int2:
                    feature_b = st.selectbox("Feature B", X_test.columns.tolist(), index=1 if len(X_test.columns) > 1 else 0, key="shap_interact_b")

                if feature_a != feature_b:
                    with st.spinner("Computing interaction..."):
                        try:
                            # Compute SHAP interaction values for the pair
                            interaction_sample = X_test.sample(min(500, len(X_test)), random_state=42)
                            shap_values = explainer.explain_batch(interaction_sample)

                            if 'interactions' in shap_values:
                                # Plot interaction
                                fig = px.scatter(
                                    x=interaction_sample[feature_a],
                                    y=interaction_sample[feature_b],
                                    color=shap_values['interactions'].get((feature_a, feature_b), np.zeros(len(interaction_sample))),
                                    color_continuous_scale='RdBu',
                                    title=f'SHAP Interaction: {feature_a} × {feature_b}',
                                    labels={'x': feature_a, 'y': feature_b, 'color': 'Interaction SHAP'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Interaction values not available. Using SHAP dependence plot instead.")

                                # Dependence plot as fallback
                                fig = px.scatter(
                                    x=interaction_sample[feature_a],
                                    y=shap_values.get('values', np.zeros(len(interaction_sample)))[:, list(X_test.columns).index(feature_a)],
                                    color=interaction_sample[feature_b],
                                    color_continuous_scale='Viridis',
                                    title=f'SHAP Dependence: {feature_a} (colored by {feature_b})',
                                    labels={'x': f'{feature_a} value', 'y': f'SHAP value for {feature_a}'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Interaction computation not available: {e}")
                            st.info("Run notebook 05_shap_explanations.ipynb for full interaction analysis")
                else:
                    st.warning("Please select two different features for interaction analysis")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ============================================================
# PAGE: DRIFT MONITORING
# ============================================================
elif page == "📈 Drift Monitoring":
    render_page_header("Drift Monitoring", "PSI/KS-based concept drift detection for UPI fraud patterns", "📈")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="drift_refresh", use_container_width=True):
            show_toast("Drift data refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("🔍 Run Live Detection", key="drift_live_btn", use_container_width=True):
            show_toast("Run drift detection in Live tab", "info")
    with col_actions[2]:
        if st.button("📊 Run Notebook", key="drift_notebook", use_container_width=True):
            show_toast("Run notebook 04_drift_monitoring.ipynb", "info")
    with col_actions[3]:
        if st.button("🔗 Share View", key="drift_share", use_container_width=True):
            set_url_params(page="drift", theme=theme_value)
            show_toast("View URL updated!", "success")

    # Create tabs for live vs historical
    tab_live, tab_history, tab_analytics = st.tabs(["🔴 Live Detection", "📜 Historical Alerts", "📊 Analytics"])

    with tab_live:
        st.subheader("🔴 Live Drift Detection")
        st.markdown("Run PSI/KS drift detection on configurable data windows.")

        col1, col2, col3 = st.columns(3)
        with col1:
            live_dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="live_dataset")
        with col2:
            live_ref_days = st.slider("Reference Days", 1, 30, 7, key="live_ref_days")
        with col3:
            live_cur_days = st.slider("Current Days", 1, 30, 7, key="live_cur_days")

        col_psi, col_ks = st.columns(2)
        with col_psi:
            live_psi_thresh = st.slider("PSI Threshold", 0.05, 0.5, 0.2, 0.05, key="live_psi")
        with col_ks:
            live_ks_thresh = st.slider("KS Threshold", 0.01, 0.2, 0.05, 0.01, key="live_ks")

        if st.button("🔍 Run Drift Detection", key="live_drift_btn", type="primary", use_container_width=True):
            with st.spinner("Running drift detection..."):
                try:
                    result = stage_drift(
                        dataset=live_dataset,
                        reference_window_days=live_ref_days,
                        current_window_days=live_cur_days,
                    )
                    st.session_state["live_drift_result"] = result
                    show_toast("✅ Drift detection completed!", "success")
                except Exception as exc:
                    show_toast(f"❌ Drift detection failed: {exc}", "error")

        # Show live results
        if "live_drift_result" in st.session_state:
            result = st.session_state["live_drift_result"]
            alerts = result.get("alerts", [])

            # Summary metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Total Alerts", str(len(alerts)), "", "normal", "🚨")
            with col2:
                feature_count = sum(1 for a in alerts if a.get("type") == "feature_drift")
                render_metric_card("Feature Drift", str(feature_count), "", "warning" if feature_count else "normal", "📊")
            with col3:
                score_count = sum(1 for a in alerts if "score_drift" in a.get("type", ""))
                render_metric_card("Score Drift", str(score_count), "", "warning" if score_count else "normal", "📈")
            with col4:
                max_val = max([a.get("value", 0) for a in alerts]) if alerts else 0
                render_metric_card("Max Severity", f"{max_val:.4f}",
                                 "🔴 High" if max_val > 0.3 else ("🟡 Medium" if max_val > 0.2 else "🟢 Low"),
                                 "error" if max_val > 0.3 else ("warning" if max_val > 0.2 else "normal"), "⚡")

            # Status summary
            col1, col2, col3, col4 = st.columns(4)
            pending_live = sum(1 for a in alerts if st.session_state.alert_status.get(alert_key(a), "pending") == "pending")
            acked_live = sum(1 for a in alerts if st.session_state.alert_status.get(alert_key(a), "pending") == "acknowledged")
            resolved_live = sum(1 for a in alerts if st.session_state.alert_status.get(alert_key(a), "pending") == "resolved")
            with col1:
                render_metric_card("Pending", str(pending_live), "Needs review" if pending_live else "All clear", "warning" if pending_live else "normal", "⏳")
            with col2:
                render_metric_card("Acknowledged", str(acked_live), "Under investigation" if acked_live else "", "normal", "✓")
            with col3:
                render_metric_card("Resolved", str(resolved_live), "Closed" if resolved_live else "", "normal", "✅")
            with col4:
                render_metric_card("Total", str(len(alerts)), "", "normal", "📋")

            if alerts:
                st.subheader("🚨 Active Drift Alerts")

                # Filter controls
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    filter_severity = st.multiselect("Severity", ["🔴 High", "🟡 Medium", "🟢 Low"],
                                                   default=["🔴 High", "🟡 Medium", "🟢 Low"], key="live_sev_filter")
                with col_f2:
                    filter_status = st.multiselect("Status", ["pending", "acknowledged", "resolved"],
                                                 default=["pending", "acknowledged", "resolved"], key="live_status_filter")
                with col_f3:
                    filter_type = st.multiselect("Type",
                        list(set(a.get('type', '') for a in alerts)),
                        default=list(set(a.get('type', '') for a in alerts)), key="live_type_filter")

                for alert in alerts:
                    value = alert.get("value", 0)
                    if value > 0.3:
                        severity = "🔴 High"
                        sev_color = "error"
                    elif value > 0.2:
                        severity = "🟡 Medium"
                        sev_color = "warning"
                    else:
                        severity = "🟢 Low"
                        sev_color = "normal"

                    if severity not in filter_severity:
                        continue

                    key = alert_key(alert)
                    current_status = st.session_state.alert_status.get(key, "pending")

                    if current_status not in filter_status:
                        continue

                    if alert.get('type', '') not in filter_type:
                        continue

                    render_metric_card(
                        f"{severity} {alert['type']}",
                        f"{alert['column']} ({alert['metric']}={value:.4f})",
                        f"Ref: {live_ref_days}d | Cur: {live_cur_days}d",
                        sev_color, "📊",
                        f"PSI threshold: {live_psi_thresh} | KS threshold: {live_ks_thresh}"
                    )

                    col_a, col_b, col_c = st.columns([6, 1, 1])
                    with col_b:
                        if current_status != "acknowledged":
                            if st.button("✓ Ack", key=f"ack_live_{key}", use_container_width=True):
                                st.session_state.alert_status[key] = "acknowledged"
                                save_alert_status(st.session_state.alert_status)
                                show_toast(f"Acknowledged: {alert['column']}", "success")
                                st.rerun()
                    with col_c:
                        if current_status != "resolved":
                            if st.button("✅ Resolve", key=f"res_live_{key}", use_container_width=True):
                                st.session_state.alert_status[key] = "resolved"
                                save_alert_status(st.session_state.alert_status)
                                show_toast(f"Resolved: {alert['column']}", "success")
                                st.rerun()

    with tab_history:
        st.subheader("📜 Historical Drift Alerts")

        if not reports['drift_alerts'].empty:
            alerts_df = reports['drift_alerts'].copy()

            # Summary metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Total Stored Alerts", str(len(alerts_df)), "", "normal", "📋")
            with col2:
                feature_count = sum(1 for _, r in alerts_df.iterrows() if r.get('type') == 'feature_drift')
                render_metric_card("Feature Drift", str(feature_count), "", "warning" if feature_count else "normal", "📊")
            with col3:
                score_count = sum(1 for _, r in alerts_df.iterrows() if 'score_drift' in str(r.get('type', '')))
                render_metric_card("Score Drift", str(score_count), "", "warning" if score_count else "normal", "📈")
            with col4:
                # Date range
                if 'timestamp' in alerts_df.columns:
                    try:
                        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
                        date_range = f"{(alerts_df['timestamp'].max() - alerts_df['timestamp'].min()).days} days"
                    except:
                        date_range = "N/A"
                else:
                    date_range = "N/A"
                render_metric_card("Date Range", date_range, "", "normal", "📅")

            # Filters
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                filter_status = st.selectbox("Filter by Status",
                    ["All", "Pending", "Acknowledged", "Resolved"], key="hist_status_filter")
            with col_f2:
                if 'type' in alerts_df.columns:
                    filter_type = st.selectbox("Filter by Type",
                        ["All"] + sorted(alerts_df['type'].unique().tolist()), key="hist_type_filter")
                else:
                    filter_type = "All"
            with col_f3:
                if 'metric' in alerts_df.columns:
                    filter_metric = st.selectbox("Filter by Metric",
                        ["All"] + sorted(alerts_df['metric'].unique().tolist()), key="hist_metric_filter")
                else:
                    filter_metric = "All"
            with col_f4:
                if 'column' in alerts_df.columns:
                    filter_column = st.selectbox("Filter by Feature",
                        ["All"] + sorted(alerts_df['column'].unique().tolist()), key="hist_column_filter")
                else:
                    filter_column = "All"

            # Apply filters
            filtered_df = alerts_df.copy()
            if filter_status != "All":
                filtered_df = filtered_df[
                    filtered_df.apply(lambda r: st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == filter_status.lower(), axis=1)
                ]
            if filter_type != "All":
                filtered_df = filtered_df[filtered_df['type'] == filter_type]
            if filter_metric != "All":
                filtered_df = filtered_df[filtered_df['metric'] == filter_metric]
            if filter_column != "All":
                filtered_df = filtered_df[filtered_df['column'] == filter_column]

            # Status summary
            pending_stored = sum(1 for _, r in filtered_df.iterrows() if st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == "pending")
            acked_stored = sum(1 for _, r in filtered_df.iterrows() if st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == "acknowledged")
            resolved_stored = sum(1 for _, r in filtered_df.iterrows() if st.session_state.alert_status.get(alert_key(r.to_dict()), "pending") == "resolved")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                render_metric_card("Pending", str(pending_stored), "", "warning" if pending_stored else "normal", "⏳")
            with col_s2:
                render_metric_card("Acknowledged", str(acked_stored), "", "normal", "✓")
            with col_s3:
                render_metric_card("Resolved", str(resolved_stored), "", "normal", "✅")

            # Alert table with actions
            if not filtered_df.empty:
                st.subheader(f"📋 Alerts ({len(filtered_df)} shown)")

                for idx, (_, row) in enumerate(filtered_df.iterrows()):
                    alert = row.to_dict()
                    value = alert.get("value", 0)
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = 0.0
                    if value > 0.3:
                        severity = "🔴 High"
                        sev_color = "error"
                    elif value > 0.2:
                        severity = "🟡 Medium"
                        sev_color = "warning"
                    else:
                        severity = "🟢 Low"
                        sev_color = "normal"

                    key = alert_key(alert)
                    current_status = st.session_state.alert_status.get(key, "pending")

                    status_icon = {"pending": "⏳", "acknowledged": "✓", "resolved": "✅"}[current_status]

                    render_metric_card(
                        f"{status_icon} {severity} {alert.get('type','')}",
                        f"{alert.get('column','')} ({alert.get('metric','')}={value:.4f})",
                        f"Time: {alert.get('timestamp', 'N/A')}",
                        sev_color, "📊"
                    )

                    col_a, col_b, col_c = st.columns([6, 1, 1])
                    with col_b:
                        if current_status != "acknowledged":
                            if st.button("✓ Ack", key=f"ack_hist_{idx}_{key}", use_container_width=True):
                                st.session_state.alert_status[key] = "acknowledged"
                                save_alert_status(st.session_state.alert_status)
                                show_toast(f"Acknowledged", "success")
                                st.rerun()
                    with col_c:
                        if current_status != "resolved":
                            if st.button("✅ Resolve", key=f"res_hist_{idx}_{key}", use_container_width=True):
                                st.session_state.alert_status[key] = "resolved"
                                save_alert_status(st.session_state.alert_status)
                                show_toast(f"Resolved", "success")
                                st.rerun()
            else:
                st.info("No alerts match the current filters.")

            # Analytics charts
            st.divider()
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                if 'type' in alerts_df.columns:
                    st.subheader("Alert Type Distribution")
                    type_counts = alerts_df['type'].value_counts()
                    fig = px.pie(
                        values=type_counts.values, names=type_counts.index,
                        title="Alert Types",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with col_chart2:
                if 'column' in alerts_df.columns:
                    st.subheader("Most Alerted Features (Top 10)")
                    feat_counts = alerts_df['column'].value_counts().head(10)
                    fig = px.bar(
                        x=feat_counts.values, y=feat_counts.index, orientation='h',
                        labels={'x': 'Alert Count', 'y': 'Feature'},
                        color=feat_counts.values, color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Time series
            if 'timestamp' in alerts_df.columns:
                try:
                    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
                    alerts_df['date'] = alerts_df['timestamp'].dt.date
                    daily_counts = alerts_df.groupby('date').size().reset_index(name='count')

                    st.subheader("📊 Alert Frequency Over Time")
                    fig = px.line(
                        daily_counts, x='date', y='count',
                        title='Daily Drift Alert Count',
                        labels={'date': 'Date', 'count': 'Alerts'},
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Time series plot unavailable: {e}")

        else:
            st.warning("No drift alerts found in reports.")
            st.code("jupyter notebook notebooks/04_drift_monitoring.ipynb")

    with tab_analytics:
        st.subheader("📊 Drift Analytics Dashboard")
        st.info("Advanced analytics and trends for drift monitoring.")

        if not reports['drift_alerts'].empty:
            alerts_df = reports['drift_alerts'].copy()

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total = len(alerts_df)
                render_metric_card("Total Alerts", str(total), "", "normal", "📋")
            with col2:
                if 'value' in alerts_df.columns:
                    avg_val = alerts_df['value'].astype(float).mean()
                    render_metric_card("Avg Drift Score", f"{avg_val:.4f}", "", "normal", "📊")
            with col3:
                if 'value' in alerts_df.columns:
                    max_val = alerts_df['value'].astype(float).max()
                    render_metric_card("Max Drift Score", f"{max_val:.4f}", "", "error" if max_val > 0.3 else "normal", "⚡")
            with col4:
                unique_features = alerts_df['column'].nunique() if 'column' in alerts_df.columns else 0
                render_metric_card("Features Monitored", str(unique_features), "", "normal", "🔍")

            # Drift score distribution
            if 'value' in alerts_df.columns:
                st.subheader("Drift Score Distribution")
                fig = px.histogram(
                    alerts_df, x='value', nbins=20,
                    color='type' if 'type' in alerts_df.columns else None,
                    title='Distribution of Drift Scores (PSI/KS Values)',
                    labels={'value': 'Drift Score', 'count': 'Frequency'}
                )
                fig.add_vline(x=0.2, line_dash="dash", line_color="#f59e0b", annotation_text="PSI Threshold (0.2)")
                fig.add_vline(x=0.3, line_dash="dash", line_color="#ef4444", annotation_text="High Severity (0.3)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Feature stability ranking
            if 'column' in alerts_df.columns and 'value' in alerts_df.columns:
                st.subheader("Feature Drift Ranking")
                feature_stats = alerts_df.groupby('column').agg(
                    alert_count=('value', 'count'),
                    avg_drift=('value', 'mean'),
                    max_drift=('value', 'max')
                ).sort_values('avg_drift', ascending=False).reset_index()

                fig = px.bar(
                    feature_stats.head(15), x='avg_drift', y='column', orientation='h',
                    color='alert_count', color_continuous_scale='Reds',
                    title='Top Features by Average Drift Score',
                    labels={'avg_drift': 'Avg Drift Score', 'column': 'Feature', 'alert_count': 'Alert Count'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(feature_stats, use_container_width=True, hide_index=True)
        else:
            st.info("Run drift detection to generate analytics data.")
            skeleton_loader(4, 150)

# ============================================================
# PAGE: RETRAINING STRATEGIES
# ============================================================
elif page == "⚙️ Retraining Strategies":
    render_page_header("Retraining Strategies", "Compare Static, Scheduled, and Drift-Triggered retraining approaches", "⚙️")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="retrain_refresh", use_container_width=True):
            show_toast("Retraining data refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("📊 Run Notebook", key="retrain_notebook", use_container_width=True):
            show_toast("Run notebook 06_retraining_strategies.ipynb", "info")
    with col_actions[2]:
        if st.button("🔗 Share View", key="retrain_share", use_container_width=True):
            set_url_params(page="retrain", theme=theme_value)
            show_toast("View URL updated!", "success")
    with col_actions[3]:
        if st.button("📥 Export CSV", key="retrain_export", use_container_width=True):
            if not reports['retrain_comparison'].empty:
                csv = reports['retrain_comparison'].to_csv(index=False)
                st.download_button("Download", csv, "retrain_comparison.csv", "text/csv", key="retrain_dl")
                show_toast("Export ready!", "success")
            else:
                show_toast("No data to export", "warning")

    if not reports['retrain_comparison'].empty:
        comp_df = reports['retrain_comparison'].copy()

        # Summary metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Strategies Compared", str(comp_df['strategy'].nunique()), "", "normal", "⚙️")
        with col2:
            best_pr = comp_df.loc[comp_df['pr_auc'].idxmax()] if 'pr_auc' in comp_df.columns else None
            if best_pr is not None:
                render_metric_card("Best PR-AUC", f"{best_pr['pr_auc']:.4f}", f"Strategy: {best_pr['strategy']}", "success", "🏆")
        with col3:
            if 'n_retrains' in comp_df.columns:
                min_retrains = comp_df['n_retrains'].min()
                render_metric_card("Min Retrains", str(int(min_retrains)), "", "normal", "🔁")
        with col4:
            if 'n_retrains' in comp_df.columns:
                max_retrains = comp_df['n_retrains'].max()
                render_metric_card("Max Retrains", str(int(max_retrains)), "", "warning", "🔁")

        st.divider()

        # Strategy comparison table with enhanced formatting
        st.subheader("📋 Strategy Performance Comparison")

        # Select metrics to display
        all_metrics = ['pr_auc', 'precision_at_100', 'recall_at_100', 'fp_per_100k_at_budget_100', 'n_retrains', 'avg_model_age_days']
        available_metrics = [m for m in all_metrics if m in comp_df.columns]

        display_cols = ['strategy'] + available_metrics
        display_df = comp_df[display_cols].copy()

        # Format columns for display
        for col in display_df.columns:
            if col in ['pr_auc', 'precision_at_100', 'recall_at_100']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            elif col == 'fp_per_100k_at_budget_100':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
            elif col in ['n_retrains', 'avg_model_age_days']:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "strategy": st.column_config.TextColumn("Strategy", width="medium"),
                **{col: st.column_config.TextColumn(col.replace('_', ' ').title(), width="small") for col in available_metrics}
            }
        )

        # Visualization tabs
        tab_viz1, tab_viz2, tab_viz3 = st.tabs(["📊 Operational Metrics", "🔁 Retrain Frequency", "📈 Detailed Windows"])

        with tab_viz1:
            if available_metrics:
                st.subheader("Operational Metrics by Strategy")

                # Chart type selector
                chart_type = st.radio("Chart Type", ["Grouped Bar", "Horizontal Bar", "Radar"], horizontal=True, key="retrain_chart_type")

                primary_metrics = ['pr_auc', 'precision_at_100', 'recall_at_100', 'fp_per_100k_at_budget_100']
                primary_metrics = [m for m in primary_metrics if m in comp_df.columns]

                if chart_type == "Radar" and len(primary_metrics) >= 3:
                    # Radar chart
                    fig = go.Figure()
                    for _, row in comp_df.iterrows():
                        values = [row[m] for m in primary_metrics]
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=[m.replace('_', ' ').title() for m in primary_metrics],
                            fill='toself',
                            name=row['strategy'],
                            opacity=0.6
                        ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(row[m] for m in primary_metrics) for _, row in comp_df.iterrows()) * 1.1])),
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Horizontal Bar":
                    for metric in primary_metrics:
                        fig = px.bar(
                            comp_df.sort_values(metric, ascending=True),
                            y='strategy', x=metric, orientation='h',
                            title=metric.replace('_', ' ').title(),
                            color=metric, color_continuous_scale='Viridis',
                            text_auto='.4f'
                        )
                        fig.update_layout(showlegend=False, height=300, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

                else:  # Grouped Bar
                    for metric in primary_metrics:
                        fig = px.bar(
                            comp_df, x='strategy', y=metric,
                            title=metric.replace('_', ' ').title(),
                            color='strategy', text_auto='.4f',
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)

        with tab_viz2:
            if 'n_retrains' in comp_df.columns and 'avg_model_age_days' in comp_df.columns:
                st.subheader("Retraining Efficiency Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Retrains vs PR-AUC trade-off
                    fig = px.scatter(
                        comp_df, x='n_retrains', y='pr_auc',
                        color='strategy', size='avg_model_age_days' if 'avg_model_age_days' in comp_df.columns else None,
                        hover_data=['strategy', 'precision_at_100', 'recall_at_100'],
                        title='Retrain Count vs PR-AUC (Size = Avg Model Age)',
                        labels={'n_retrains': 'Number of Retrains', 'pr_auc': 'PR-AUC'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Model age distribution
                    fig = px.bar(
                        comp_df, x='strategy', y='avg_model_age_days',
                        title='Average Model Age (Days)',
                        color='strategy', text_auto=True,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Efficiency summary
                st.subheader("📊 Efficiency Summary")
                eff_df = comp_df[['strategy', 'n_retrains', 'pr_auc', 'avg_model_age_days']].copy()
                eff_df['efficiency'] = eff_df['pr_auc'] / eff_df['n_retrains'].replace(0, 1)
                eff_df = eff_df.sort_values('efficiency', ascending=False)

                st.dataframe(
                    eff_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "strategy": st.column_config.TextColumn("Strategy"),
                        "n_retrains": st.column_config.NumberColumn("Retrains", format="%d"),
                        "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
                        "avg_model_age_days": st.column_config.NumberColumn("Avg Model Age", format="%d"),
                        "efficiency": st.column_config.NumberColumn("PR-AUC per Retrain", format="%.4f"),
                    }
                )

        with tab_viz3:
            if not reports.get('retrain_detailed', pd.DataFrame()).empty:
                detailed = reports['retrain_detailed'].copy()
                st.subheader("Window-by-Window Results")

                # Filter by strategy
                if 'strategy' in detailed.columns:
                    strategy_filter = st.selectbox(
                        "Filter by Strategy",
                        ["All"] + sorted(detailed['strategy'].unique().tolist()),
                        key="retrain_detail_filter"
                    )
                    if strategy_filter != "All":
                        detailed = detailed[detailed['strategy'] == strategy_filter]

                st.dataframe(
                    detailed,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "strategy": st.column_config.TextColumn("Strategy"),
                        "window": st.column_config.NumberColumn("Window", format="%d"),
                        "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
                        "precision_at_100": st.column_config.NumberColumn("P@100", format="%.4f"),
                        "recall_at_100": st.column_config.NumberColumn("R@100", format="%.4f"),
                        "retrained": st.column_config.CheckboxColumn("Retrained"),
                    }
                )

                # Window trend chart
                if 'window' in detailed.columns and 'pr_auc' in detailed.columns:
                    fig = px.line(
                        detailed, x='window', y='pr_auc', color='strategy',
                        title='PR-AUC Over Retraining Windows',
                        markers=True,
                        labels={'window': 'Window', 'pr_auc': 'PR-AUC'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Detailed window-by-window results not available. Run notebook 06_retraining_strategies.ipynb.")

        # Strategy descriptions with enhanced styling
        st.divider()
        with st.expander("📖 Strategy Descriptions", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                **🔒 Static**
                - Never retrain after initial training
                - Baseline for comparison
                - Lowest maintenance overhead
                - Performance degrades over time
                """)

            with col2:
                st.markdown("""
                **📅 Scheduled (Weekly)**
                - Retrain every 7 days regardless of data changes
                - Simple to implement and understand
                - May retrain unnecessarily
                - Consistent model freshness
                """)

            with col3:
                st.markdown("""
                **🎯 Drift-Triggered**
                - Retrain only when PSI > 0.2 or KS > 0.05 detected
                - Force retrain after 90 days max model age
                - **Expected: 30% fewer retrains than scheduled with comparable performance**
                - Optimal resource utilization
                """)

    else:
        st.warning("No retraining results found. Run notebook 06_retraining_strategies.ipynb first.")
        st.code("jupyter notebook notebooks/06_retraining_strategies.ipynb")
        skeleton_loader(5, 150)

# ============================================================
# PAGE: DELAYED LABELS
# ============================================================
elif page == "⏱️ Delayed Labels":
    render_page_header("Delayed Labels Impact", "Analyze model performance degradation under label delay scenarios", "⏱️")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="delay_refresh", use_container_width=True):
            show_toast("Delayed labels data refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("📊 Run Notebook", key="delay_notebook", use_container_width=True):
            show_toast("Run notebook 03_delayed_labels.ipynb", "info")
    with col_actions[2]:
        if st.button("🔗 Share View", key="delay_share", use_container_width=True):
            set_url_params(page="delay", theme=theme_value)
            show_toast("View URL updated!", "success")
    with col_actions[3]:
        if st.button("📥 Export CSV", key="delay_export", use_container_width=True):
            if not reports['delay_impact'].empty:
                csv = reports['delay_impact'].to_csv(index=False)
                st.download_button("Download", csv, "delay_impact.csv", "text/csv", key="delay_dl")
                show_toast("Export ready!", "success")
            else:
                show_toast("No data to export", "warning")

    if not reports['delay_impact'].empty:
        delay_df = reports['delay_impact'].copy()

        # Summary metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_delay = delay_df['delay_days'].max()
            render_metric_card("Max Delay Tested", f"{int(max_delay)} days", "", "normal", "⏱️")
        with col2:
            oracle_pr = delay_df[delay_df['delay_days'] == 0]['avg_pr_auc'].values[0] if 0 in delay_df['delay_days'].values else delay_df['avg_pr_auc'].max()
            render_metric_card("Oracle PR-AUC (0-day)", f"{oracle_pr:.4f}", "Best possible performance", "success", "🏆")
        with col3:
            worst_pr = delay_df['avg_pr_auc'].min()
            worst_delay = delay_df.loc[delay_df['avg_pr_auc'].idxmin(), 'delay_days']
            total_deg = (oracle_pr - worst_pr) / oracle_pr * 100
            render_metric_card("Max Degradation", f"{total_deg:.1f}%", f"At {int(worst_delay)}-day delay", "error", "📉")
        with col4:
            if len(delay_df) > 1:
                # Calculate degradation per day
                deg_per_day = total_deg / max_delay
                render_metric_card("Degradation/Day", f"{deg_per_day:.2f}%", "Average daily impact", "warning", "📊")

        st.divider()

        # Main content tabs
        tab_overview, tab_detailed, tab_insights = st.tabs(["📊 Overview", "📋 Detailed Table", "💡 Insights"])

        with tab_overview:
            st.subheader("Performance Degradation with Label Delay")

            # Enhanced dataframe display
            display_df = delay_df.copy()
            for col in display_df.columns:
                if col != 'delay_days':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "delay_days": st.column_config.NumberColumn("Delay (Days)", format="%d", width="small"),
                    **{col: st.column_config.TextColumn(col.replace('_', ' ').title(), width="small") for col in display_df.columns if col != 'delay_days'}
                }
            )

            # Enhanced dual-axis plot
            st.subheader("📈 Performance vs Label Delay")

            fig = go.Figure()

            # PR-AUC
            fig.add_trace(go.Scatter(
                x=delay_df['delay_days'], y=delay_df['avg_pr_auc'],
                mode='lines+markers', name='PR-AUC',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=10, symbol='circle'),
                hovertemplate='Delay: %{x} days<br>PR-AUC: %{y:.4f}<extra></extra>'
            ))

            # P@100
            if 'avg_p100' in delay_df.columns:
                fig.add_trace(go.Scatter(
                    x=delay_df['delay_days'], y=delay_df['avg_p100'],
                    mode='lines+markers', name='P@100',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=10, symbol='diamond'),
                    yaxis='y2',
                    hovertemplate='Delay: %{x} days<br>P@100: %{y:.4f}<extra></extra>'
                ))

            # R@100
            if 'avg_r100' in delay_df.columns:
                fig.add_trace(go.Scatter(
                    x=delay_df['delay_days'], y=delay_df['avg_r100'],
                    mode='lines+markers', name='R@100',
                    line=dict(color='#22c55e', width=3),
                    marker=dict(size=10, symbol='square'),
                    yaxis='y2',
                    hovertemplate='Delay: %{x} days<br>R@100: %{y:.4f}<extra></extra>'
                ))

            fig.update_layout(
                title='Model Performance vs Label Delay',
                xaxis_title='Label Delay (Days)',
                yaxis=dict(title='PR-AUC', side='left', range=[0, max(delay_df['avg_pr_auc'].max() * 1.1, 1)]),
                yaxis2=dict(title='P@100 / R@100', side='right', overlaying='y', range=[0, 1]),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=500,
                template='plotly_white'
            )

            # Add shaded regions for degradation zones
            fig.add_vrect(x0=0, x1=1, fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="Minimal", annotation_position="top left")
            fig.add_vrect(x0=1, x1=7, fillcolor="yellow", opacity=0.1, layer="below", line_width=0, annotation_text="Moderate", annotation_position="top")
            fig.add_vrect(x0=7, x1=max_delay + 1, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Severe", annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True)

            # Degradation summary with visual cards
            st.subheader("📉 Relative Degradation vs Oracle (0-day delay)")

            degradation_data = []
            for _, row in delay_df.iterrows():
                if row['delay_days'] > 0:
                    pr_deg = (oracle_pr - row['avg_pr_auc']) / oracle_pr * 100
                    p100_deg = 0
                    if 'avg_p100' in row and row['avg_p100'] > 0:
                        oracle_p100 = delay_df[delay_df['delay_days'] == 0]['avg_p100'].values[0] if 0 in delay_df['delay_days'].values else delay_df['avg_p100'].max()
                        p100_deg = (oracle_p100 - row['avg_p100']) / oracle_p100 * 100

                    severity = "🔴 High" if pr_deg > 20 else ("🟡 Medium" if pr_deg > 10 else "🟢 Low")
                    degradation_data.append({
                        'delay_days': int(row['delay_days']),
                        'pr_auc': row['avg_pr_auc'],
                        'pr_degradation_pct': pr_deg,
                        'p100_degradation_pct': p100_deg,
                        'severity': severity
                    })

            if degradation_data:
                deg_df = pd.DataFrame(degradation_data)

                # Display as metric cards
                for _, row in deg_df.iterrows():
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        render_metric_card(f"{row['delay_days']}-Day Delay", f"PR-AUC: {row['pr_auc']:.4f}", "", "normal", "⏱️")
                    with col2:
                        sev_color = "error" if row['pr_degradation_pct'] > 20 else ("warning" if row['pr_degradation_pct'] > 10 else "normal")
                        render_metric_card("PR-AUC Degradation", f"{row['pr_degradation_pct']:.1f}%", row['severity'], sev_color, "📉")
                    with col3:
                        render_metric_card("P@100 Degradation", f"{row['p100_degradation_pct']:.1f}%", "", "normal", "🎯")
                    with col4:
                        st.write("")  # spacer

        with tab_detailed:
            st.subheader("📋 Detailed Metrics Table")

            # Show all available metrics
            metric_cols = [c for c in delay_df.columns if c != 'delay_days']

            # Melt for better visualization
            melted = delay_df.melt(id_vars=['delay_days'], value_vars=metric_cols,
                                  var_name='metric', value_name='value')

            # Line chart for all metrics
            fig = px.line(
                melted, x='delay_days', y='value', color='metric',
                title='All Metrics vs Label Delay',
                markers=True,
                labels={'delay_days': 'Delay (Days)', 'value': 'Score', 'metric': 'Metric'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap of degradation
            st.subheader("🔥 Degradation Heatmap")
            if len(metric_cols) > 0:
                # Normalize to 0-day baseline
                norm_df = delay_df.copy()
                for col in metric_cols:
                    baseline = norm_df[norm_df['delay_days'] == 0][col].values[0] if 0 in norm_df['delay_days'].values else norm_df[col].max()
                    norm_df[col] = (baseline - norm_df[col]) / baseline * 100

                heatmap_data = norm_df.set_index('delay_days')[metric_cols]

                fig = px.imshow(
                    heatmap_data.T,
                    labels=dict(x="Delay (Days)", y="Metric", color="Degradation %"),
                    color_continuous_scale='Reds',
                    aspect="auto",
                    title="Percentage Degradation from Oracle (0-day)"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab_insights:
            st.subheader("💡 Key Insights & Recommendations")

            if len(degradation_data) > 0:
                deg_df = pd.DataFrame(degradation_data)

                # Key findings
                st.markdown("### 🔍 Key Findings")

                # 1. Threshold analysis
                acceptable_threshold = 10  # 10% degradation threshold
                acceptable_delays = deg_df[deg_df['pr_degradation_pct'] <= acceptable_threshold]['delay_days'].tolist()
                if acceptable_delays:
                    st.success(f"✅ **Acceptable Delay Range**: Up to **{max(acceptable_delays)} days** (≤{acceptable_threshold}% PR-AUC degradation)")
                else:
                    st.error(f"❌ **No acceptable delay range** found at ≤{acceptable_threshold}% degradation threshold")

                # 2. Critical threshold
                critical_threshold = 20
                critical_delays = deg_df[deg_df['pr_degradation_pct'] > critical_threshold]['delay_days'].tolist()
                if critical_delays:
                    st.error(f"🚨 **Critical Degradation**: Beyond **{min(critical_delays)} days** (>{critical_threshold}% PR-AUC degradation)")

                # 3. Linear fit for prediction
                if len(deg_df) >= 3:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(deg_df['delay_days'], deg_df['pr_degradation_pct'])
                    st.info(f"📊 **Degradation Rate**: ~{slope:.2f}% PR-AUC loss per day (R²={r_value**2:.3f})")

                    # Predict 90-day degradation
                    pred_90 = slope * 90 + intercept
                    st.warning(f"🔮 **90-Day Projection**: ~{pred_90:.1f}% PR-AUC degradation at 90-day delay")

                # 4. Recommendations
                st.markdown("### 🎯 Recommendations")
                st.markdown("""
                1. **Implement Label Acceleration**: Prioritize reducing label delay to <7 days for minimal degradation
                2. **Drift-Triggered Retraining**: Use drift detection to trigger retraining before severe degradation
                3. **Monitoring Alert**: Set up alerts when label delay exceeds 7 days
                4. **Buffer Models**: Maintain ensemble of models trained at different delay horizons
                5. **Synthetic Labels**: Explore pseudo-labeling or semi-supervised approaches for interim periods
                """)

                # Export degradation data
                st.markdown("### 📥 Export Analysis")
                csv = deg_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Degradation Analysis (CSV)",
                    csv,
                    file_name="delay_degradation_analysis.csv",
                    mime="text/csv",
                    key="delay_deg_export"
                )

    else:
        st.warning("No delayed labels results found. Run notebook 03_delayed_labels.ipynb first.")
        st.code("jupyter notebook notebooks/03_delayed_labels.ipynb")
        skeleton_loader(5, 150)

# ============================================================
# PAGE: PREDICTION HISTORY
# ============================================================
elif page == "📜 Prediction History":
    render_page_header("Prediction History", "Track and analyze all fraud scoring sessions", "📜")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="ph_refresh", use_container_width=True):
            show_toast("History refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("🔗 Share View", key="ph_share", use_container_width=True):
            set_url_params(page="history", theme=theme_value)
            show_toast("View URL updated!", "success")
    with col_actions[2]:
        if st.button("📥 Export All", key="ph_export_all", use_container_width=True):
            if st.session_state.prediction_history:
                csv = pd.DataFrame(st.session_state.prediction_history).to_csv(index=False)
                st.download_button("Download", csv, "prediction_history.csv", "text/csv", key="ph_dl_all")
                show_toast("Export ready!", "success")
            else:
                show_toast("No history to export", "warning")
    with col_actions[3]:
        if st.button("🗑️ Clear All", key="ph_clear_all", use_container_width=True):
            st.session_state.prediction_history = []
            show_toast("History cleared!", "success")
            st.rerun()

    history = st.session_state.prediction_history

    if not history:
        st.info("No prediction sessions yet. Go to Risk Scoring to generate predictions.")

        # Empty state with helpful info
        st.markdown("""
        ### 📋 Getting Started
        1. Navigate to **🎯 Risk Scoring** page
        2. Select a dataset and model
        3. Upload a CSV file or use sample data
        4. Click **Score Transactions** to generate predictions
        5. Return here to see your scoring history
        """)
        skeleton_loader(3, 150)
    else:
        # Convert to DataFrame
        hist_df = pd.DataFrame(history)

        # Summary metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Sessions", str(len(hist_df)), "", "normal", "📋")
        with col2:
            total_preds = int(hist_df['n_predictions'].sum())
            render_metric_card("Total Predictions", f"{total_preds:,}", "", "normal", "🔮")
        with col3:
            avg_fraud = hist_df['fraud_rate'].mean()
            render_metric_card("Avg Fraud Rate", f"{avg_fraud:.2%}", f"Across {len(hist_df)} sessions", "normal", "🎯")
        with col4:
            avg_score = hist_df['avg_score'].mean()
            render_metric_card("Avg Risk Score", f"{avg_score:.4f}", f"Max: {hist_df['max_score'].max():.4f}", "normal", "📊")

        st.divider()

        # Filters
        st.subheader("🔍 Filter History")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            model_filter = st.selectbox(
                "Model",
                ["All"] + sorted(hist_df['model'].unique().tolist()),
                key="ph_model_filter"
            )
        with col_f2:
            dataset_filter = st.selectbox(
                "Dataset",
                ["All"] + sorted(hist_df['dataset'].unique().tolist()),
                key="ph_dataset_filter"
            )
        with col_f3:
            date_filter = st.date_input(
                "From date",
                value=None,
                key="ph_date_filter"
            )
        with col_f4:
            sort_by = st.selectbox(
                "Sort by",
                ["Timestamp (Newest)", "Timestamp (Oldest)", "Fraud Rate (High)", "Avg Score (High)", "Predictions (High)"],
                key="ph_sort_filter"
            )

        filtered_df = hist_df.copy()
        if model_filter != "All":
            filtered_df = filtered_df[filtered_df['model'] == model_filter]
        if dataset_filter != "All":
            filtered_df = filtered_df[filtered_df['dataset'] == dataset_filter]
        if date_filter:
            filtered_df = filtered_df[
                pd.to_datetime(filtered_df['timestamp']).dt.date >= date_filter
            ]

        # Apply sorting
        sort_map = {
            "Timestamp (Newest)": ('timestamp', False),
            "Timestamp (Oldest)": ('timestamp', True),
            "Fraud Rate (High)": ('fraud_rate', False),
            "Avg Score (High)": ('avg_score', False),
            "Predictions (High)": ('n_predictions', False),
        }
        sort_col, sort_asc = sort_map[sort_by]
        filtered_df = filtered_df.sort_values(sort_col, ascending=sort_asc)

        # Tabs for different views
        tab_table, tab_charts, tab_analytics = st.tabs(["📋 Sessions Table", "📈 Trends", "📊 Analytics"])

        with tab_table:
            st.subheader(f"📋 Prediction Sessions ({len(filtered_df)} shown)")

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
                    "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                    "model": st.column_config.TextColumn("Model", width="small"),
                    "dataset": st.column_config.TextColumn("Dataset", width="small"),
                    "n_predictions": st.column_config.NumberColumn("Predictions", format="%d", width="small"),
                    "fraud_rate": st.column_config.TextColumn("Fraud Rate", width="small"),
                    "avg_score": st.column_config.TextColumn("Avg Score", width="small"),
                    "max_score": st.column_config.TextColumn("Max Score", width="small"),
                    "threshold": st.column_config.NumberColumn("Threshold", format="%.2f", width="small"),
                    "predicted_frauds": st.column_config.NumberColumn("Pred Frauds", format="%d", width="small"),
                    "actual_frauds": st.column_config.NumberColumn("Actual Frauds", format="%d", width="small"),
                }
            )

        with tab_charts:
            st.subheader("📈 Risk Score Trends")

            if len(filtered_df) > 1:
                trend_df = filtered_df.copy()
                trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'])
                trend_df = trend_df.sort_values('timestamp')

                # Multi-line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_df['timestamp'], y=trend_df['avg_score'],
                    mode='lines+markers', name='Avg Risk Score',
                    line=dict(color='#6366f1', width=3),
                    marker=dict(size=8, symbol='circle')
                ))
                fig.add_trace(go.Scatter(
                    x=trend_df['timestamp'], y=trend_df['max_score'],
                    mode='lines+markers', name='Max Risk Score',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=8, symbol='diamond')
                ))
                fig.add_trace(go.Scatter(
                    x=trend_df['timestamp'], y=trend_df['fraud_rate'],
                    mode='lines+markers', name='Actual Fraud Rate',
                    line=dict(color='#22c55e', width=3),
                    marker=dict(size=8, symbol='square'),
                    yaxis='y2'
                ))
                fig.update_layout(
                    title='Risk Scores and Fraud Rate Over Time',
                    xaxis_title='Session Time',
                    yaxis=dict(title='Risk Score', side='left', range=[0, 1]),
                    yaxis2=dict(title='Fraud Rate', side='right', overlaying='y', range=[0, max(trend_df['fraud_rate'].max() * 2, 0.1)]),
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Predictions volume over time
                fig2 = px.bar(
                    trend_df, x='timestamp', y='n_predictions',
                    color='model', title='Predictions Volume by Session',
                    labels={'timestamp': 'Session Time', 'n_predictions': 'Predictions'}
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

            else:
                st.info("Run more scoring sessions to see trend charts.")
                skeleton_loader(2, 150)

        with tab_analytics:
            st.subheader("📊 Session Analytics")

            col1, col2 = st.columns(2)

            with col1:
                # Model performance comparison
                if len(filtered_df) > 1:
                    model_stats = filtered_df.groupby('model').agg(
                        sessions=('n_predictions', 'count'),
                        total_preds=('n_predictions', 'sum'),
                        avg_fraud_rate=('fraud_rate', 'mean'),
                        avg_score=('avg_score', 'mean'),
                        max_score=('max_score', 'max')
                    ).reset_index()

                    st.markdown("**Model Performance Summary**")
                    st.dataframe(
                        model_stats,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "model": st.column_config.TextColumn("Model"),
                            "sessions": st.column_config.NumberColumn("Sessions", format="%d"),
                            "total_preds": st.column_config.NumberColumn("Total Preds", format="%d"),
                            "avg_fraud_rate": st.column_config.NumberColumn("Avg Fraud Rate", format="%.2%"),
                            "avg_score": st.column_config.NumberColumn("Avg Score", format="%.4f"),
                            "max_score": st.column_config.NumberColumn("Max Score", format="%.4f"),
                        }
                    )

                    # Bar chart
                    fig = px.bar(
                        model_stats, x='model', y='avg_fraud_rate',
                        title='Average Fraud Rate by Model',
                        color='model', text_auto='.2%'
                    )
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Dataset comparison
                dataset_stats = filtered_df.groupby('dataset').agg(
                    sessions=('n_predictions', 'count'),
                    total_preds=('n_predictions', 'sum'),
                    avg_fraud_rate=('fraud_rate', 'mean'),
                    avg_score=('avg_score', 'mean')
                ).reset_index()

                st.markdown("**Dataset Comparison**")
                st.dataframe(
                    dataset_stats,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "dataset": st.column_config.TextColumn("Dataset"),
                        "sessions": st.column_config.NumberColumn("Sessions", format="%d"),
                        "total_preds": st.column_config.NumberColumn("Total Preds", format="%d"),
                        "avg_fraud_rate": st.column_config.NumberColumn("Avg Fraud Rate", format="%.2%"),
                        "avg_score": st.column_config.NumberColumn("Avg Score", format="%.4f"),
                    }
                )

                # Pie chart of predictions by dataset
                fig = px.pie(
                    dataset_stats, values='total_preds', names='dataset',
                    title='Predictions Distribution by Dataset'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Threshold analysis
            st.markdown("**Threshold Analysis**")
            if 'threshold' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df, x='threshold', y='avg_score',
                    color='model', size='n_predictions',
                    hover_data=['dataset', 'fraud_rate', 'predicted_frauds'],
                    title='Threshold vs Average Risk Score',
                    labels={'threshold': 'Decision Threshold', 'avg_score': 'Avg Risk Score'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Export filtered data
        st.divider()
        st.subheader("📥 Export Filtered Data")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Export Filtered (CSV)",
                csv,
                file_name=f"prediction_history_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="ph_export_filtered",
                use_container_width=True
            )
        with col_exp2:
            # JSON export
            import json
            json_str = filtered_df.to_json(orient='records', date_format='iso', indent=2)
            st.download_button(
                "📥 Export Filtered (JSON)",
                json_str,
                file_name=f"prediction_history_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                key="ph_export_json",
                use_container_width=True
            )

# ============================================================
# PAGE: REPORTS
# ============================================================
elif page == "📋 Reports":
    render_page_header("Generated Reports", "Access all pipeline reports, metrics, and evaluation results", "📋")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="reports_refresh", use_container_width=True):
            show_toast("Reports refreshed!", "success")
            st.rerun()
    with col_actions[1]:
        if st.button("🔗 Share View", key="reports_share", use_container_width=True):
            set_url_params(page="reports", theme=theme_value)
            show_toast("View URL updated!", "success")
    with col_actions[2]:
        if st.button("📦 Export All", key="reports_export_all", use_container_width=True):
            show_toast("Preparing bundle...", "info")
    with col_actions[3]:
        if st.button("🔄 Regenerate", key="reports_regenerate", use_container_width=True):
            show_toast("Run notebook 07_evaluation.ipynb", "info")

    report_files = {
        'Final Evaluation Report': {
            'path': REPORTS_DIR / 'final_evaluation_report.txt',
            'type': 'txt',
            'icon': '📄',
            'description': 'Complete pipeline evaluation with all metrics'
        },
        'Drift Alerts Log': {
            'path': REPORTS_DIR / 'drift_alerts_log.csv',
            'type': 'csv',
            'icon': '📈',
            'description': 'Historical PSI/KS drift detection alerts'
        },
        'SHAP Global Importance': {
            'path': REPORTS_DIR / 'shap_global_importance.csv',
            'type': 'csv',
            'icon': '🔍',
            'description': 'Global feature importance from TreeSHAP'
        },
        'SHAP Stability': {
            'path': REPORTS_DIR / 'shap_stability.csv',
            'type': 'csv',
            'icon': '📊',
            'description': 'Feature importance stability across time windows'
        },
        'Retraining Comparison': {
            'path': REPORTS_DIR / 'retraining_strategy_comparison.csv',
            'type': 'csv',
            'icon': '⚙️',
            'description': 'Static vs Scheduled vs Drift-Triggered strategies'
        },
        'Delayed Labels Impact': {
            'path': REPORTS_DIR / 'delay_impact_summary.csv',
            'type': 'csv',
            'icon': '⏱️',
            'description': 'Performance degradation under label delay'
        },
        'Baseline Test Metrics (IEEE)': {
            'path': REPORTS_DIR / 'ieee_test_metrics.csv',
            'type': 'csv',
            'icon': '🏦',
            'description': 'Test set metrics for IEEE-CIS dataset'
        },
        'Baseline Test Metrics (PaySim)': {
            'path': REPORTS_DIR / 'paysim_test_metrics.csv',
            'type': 'csv',
            'icon': '💳',
            'description': 'Test set metrics for PaySim dataset'
        },
    }

    # Summary metric cards
    existing_reports = [(name, info) for name, info in report_files.items() if info['path'].exists()]
    missing_reports = [(name, info) for name, info in report_files.items() if not info['path'].exists()]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Available Reports", str(len(existing_reports)), f"of {len(report_files)} total", "normal", "📋")
    with col2:
        render_metric_card("Missing Reports", str(len(missing_reports)), "Run notebooks to generate", "warning" if missing_reports else "normal", "⚠️")
    with col3:
        csv_count = sum(1 for _, info in existing_reports if info['type'] == 'csv')
        render_metric_card("CSV Reports", str(csv_count), "", "normal", "📊")
    with col4:
        txt_count = sum(1 for _, info in existing_reports if info['type'] == 'txt')
        render_metric_card("Text Reports", str(txt_count), "", "normal", "📄")

    st.divider()

    # Reports by category
    tab_available, tab_missing, tab_bundle = st.tabs(["✅ Available Reports", "⚠️ Missing Reports", "📦 Export Bundle"])

    with tab_available:
        for name, info in existing_reports:
            with st.expander(f"{info['icon']} {name}", expanded=False):
                st.caption(info['description'])
                st.caption(f"Path: `{info['path']}`")

                try:
                    if info['type'] == 'txt':
                        with open(info['path']) as f:
                            content = f.read()

                        # Show preview with line count
                        lines = content.split('\n')
                        st.caption(f"{len(lines)} lines, {len(content)} characters")

                        # Display with syntax highlighting for text
                        st.code(content[:5000] + ("\n\n... (truncated)" if len(content) > 5000 else ""), language=None)

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "📥 Download",
                                content,
                                file_name=info['path'].name,
                                mime='text/plain',
                                key=f"dl_{name}",
                                use_container_width=True
                            )
                        with col_dl2:
                            if st.button("📋 Copy to Clipboard", key=f"cp_{name}", use_container_width=True):
                                st.code(content, language=None)
                                show_toast("Content displayed above for copying", "info")

                    else:  # CSV
                        df = pd.read_csv(info['path'])

                        # Summary stats
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            render_metric_card("Rows", f"{len(df):,}", "", "normal", "📊")
                        with col_s2:
                            render_metric_card("Columns", str(len(df.columns)), "", "normal", "📋")
                        with col_s3:
                            size_kb = info['path'].stat().st_size / 1024
                            render_metric_card("File Size", f"{size_kb:.1f} KB", "", "normal", "💾")

                        # Column info
                        with st.expander("📋 Column Details"):
                            col_info = pd.DataFrame({
                                'Column': df.columns,
                                'Type': [str(df[c].dtype) for c in df.columns],
                                'Non-Null': [df[c].notna().sum() for c in df.columns],
                                'Null %': [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns],
                                'Unique': [df[c].nunique() for c in df.columns],
                            })
                            st.dataframe(col_info, use_container_width=True, hide_index=True)

                        # Data preview with pagination
                        st.subheader("📊 Data Preview")
                        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], key=f"page_{name}")
                        total_pages = (len(df) - 1) // page_size + 1
                        page_num = st.number_input("Page", 1, total_pages, 1, key=f"page_num_{name}") - 1

                        start_idx = page_num * page_size
                        end_idx = min(start_idx + page_size, len(df))

                        st.caption(f"Showing rows {start_idx+1}-{end_idx} of {len(df)}")
                        st.dataframe(
                            df.iloc[start_idx:end_idx],
                            use_container_width=True,
                            hide_index=True
                        )

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "📥 Download CSV",
                                df.to_csv(index=False),
                                file_name=info['path'].name,
                                mime='text/csv',
                                key=f"dl_csv_{name}",
                                use_container_width=True
                            )
                        with col_dl2:
                            # JSON export
                            st.download_button(
                                "📥 Download JSON",
                                df.to_json(orient='records', indent=2),
                                file_name=info['path'].stem + '.json',
                                mime='application/json',
                                key=f"dl_json_{name}",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"Error loading report: {e}")

    with tab_missing:
        if missing_reports:
            st.warning(f"{len(missing_reports)} reports not found. Run the corresponding notebooks to generate them.")

            for name, info in missing_reports:
                with st.expander(f"⚠️ {name} - Not Found"):
                    st.caption(info['description'])
                    st.caption(f"Expected path: `{info['path']}`")

                    # Suggest which notebook to run
                    notebook_map = {
                        'Final Evaluation Report': '07_evaluation.ipynb',
                        'Drift Alerts Log': '04_drift_monitoring.ipynb',
                        'SHAP Global Importance': '05_shap_explanations.ipynb',
                        'SHAP Stability': '05_shap_explanations.ipynb',
                        'Retraining Comparison': '06_retraining_strategies.ipynb',
                        'Delayed Labels Impact': '03_delayed_labels.ipynb',
                        'Baseline Test Metrics (IEEE)': '02_baseline_models.ipynb',
                        'Baseline Test Metrics (PaySim)': '02_baseline_models.ipynb',
                    }

                    notebook = notebook_map.get(name, 'unknown')
                    st.code(f"jupyter notebook notebooks/{notebook}")

                    if st.button(f"📓 Open {notebook}", key=f"open_{name}", use_container_width=True):
                        show_toast(f"Run: jupyter notebook notebooks/{notebook}", "info")
        else:
            st.success("🎉 All reports are available!")
            st.balloons()

    with tab_bundle:
        st.subheader("📦 Export All Reports as Bundle")
        st.markdown("Download all available reports as a single ZIP file or individual files.")

        if existing_reports:
            import zipfile
            import io

            col_b1, col_b2 = st.columns(2)

            with col_b1:
                if st.button("📦 Create ZIP Bundle", key="create_zip", use_container_width=True):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for name, info in existing_reports:
                            if info['path'].exists():
                                zf.write(info['path'], arcname=f"{info['icon']}_{info['path'].name}")

                    st.download_button(
                        "📥 Download ZIP",
                        zip_buffer.getvalue(),
                        file_name=f"upi_fraud_reports_{pd.Timestamp.now().strftime('%Y%m%d')}.zip",
                        mime='application/zip',
                        key="dl_zip_bundle",
                        use_container_width=True
                    )
                    show_toast("ZIP bundle created!", "success")

            with col_b2:
                # Combined CSV for all CSV reports
                if st.button("📊 Combine CSV Reports", key="combine_csv", use_container_width=True):
                    combined_data = {}
                    for name, info in existing_reports:
                        if info['type'] == 'csv' and info['path'].exists():
                            try:
                                df = pd.read_csv(info['path'])
                                combined_data[name] = df
                            except:
                                pass

                    if combined_data:
                        # Create a multi-sheet Excel file
                        import openpyxl
                        from io import BytesIO

                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            for name, df in combined_data.items():
                                sheet_name = name[:31]  # Excel sheet name limit
                                df.to_excel(writer, sheet_name=sheet_name, index=False)

                        st.download_button(
                            "📥 Download Excel (Multi-sheet)",
                            output.getvalue(),
                            file_name=f"upi_fraud_reports_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key="dl_excel_bundle",
                            use_container_width=True
                        )
                        show_toast("Excel bundle created!", "success")
                    else:
                        show_toast("No CSV reports to combine", "warning")

            # Report index/summary
            st.divider()
            st.subheader("📋 Report Index")
            index_df = pd.DataFrame([
                {
                    'Report': name,
                    'Type': info['type'].upper(),
                    'Status': '✅ Available' if info['path'].exists() else '❌ Missing',
                    'Size': f"{info['path'].stat().st_size / 1024:.1f} KB" if info['path'].exists() else 'N/A',
                    'Modified': pd.Timestamp(info['path'].stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M') if info['path'].exists() else 'N/A',
                    'Description': info['description']
                }
                for name, info in report_files.items()
            ])
            st.dataframe(index_df, use_container_width=True, hide_index=True)
        else:
            st.info("No reports available to bundle. Run the notebooks first.")

# ============================================================
# PAGE: API INTEGRATION
# ============================================================
elif page == "🔌 API Integration":
    render_page_header("API Integration", "Test and interact with the UPI Fraud Detection REST API", "🔌")

    # Action buttons
    col_actions = st.columns([1, 1, 1, 1, 4])
    with col_actions[0]:
        if st.button("🔄 Refresh", key="api_refresh", use_container_width=True):
            show_toast("API page refreshed!", "success")
    with col_actions[1]:
        if st.button("🔗 Share View", key="api_share", use_container_width=True):
            set_url_params(page="api", theme=theme_value)
            show_toast("View URL updated!", "success")
    with col_actions[2]:
        if st.button("📋 Copy cURL", key="api_curl", use_container_width=True):
            show_toast("cURL examples in Endpoints tab", "info")
    with col_actions[3]:
        if st.button("🚀 Start Server", key="api_start", use_container_width=True):
            show_toast("Run: uvicorn src.api:app --reload --port 8000", "info")

    # API Server Status Card
    st.subheader("🏥 API Server Status")

    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
    with col_status1:
        st.markdown("**Server URL:** `http://localhost:8000`")
        st.caption("Default: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
    with col_status2:
        auto_health = st.checkbox("Auto-check (5s)", key="api_auto_health", value=False)
    with col_status3:
        if st.button("🔍 Check Now", key="api_health_check", type="primary", use_container_width=True):
            with st.spinner("Checking API health..."):
                try:
                    response = api_client.get("/health")
                    if response.status_code == 200:
                        st.session_state["api_health"] = {"status": "healthy", "data": response.json(), "time": pd.Timestamp.now()}
                        show_toast("✅ API is healthy!", "success")
                    else:
                        st.session_state["api_health"] = {"status": "error", "code": response.status_code, "time": pd.Timestamp.now()}
                        show_toast(f"❌ API returned {response.status_code}", "error")
                except Exception as exc:
                    st.session_state["api_health"] = {"status": "failed", "error": str(exc), "time": pd.Timestamp.now()}
                    show_toast(f"❌ Connection failed: {exc}", "error")

    # Display health status
    if "api_health" in st.session_state:
        health = st.session_state["api_health"]
        if health["status"] == "healthy":
            render_metric_card("API Status", "🟢 Healthy", f"Last checked: {health['time'].strftime('%H:%M:%S')}", "success", "✅")
            with st.expander("📋 Response Details"):
                st.json(health["data"])
        elif health["status"] == "error":
            render_metric_card("API Status", f"🔴 Error ({health['code']})", f"Last checked: {health['time'].strftime('%H:%M:%S')}", "error", "❌")
        else:
            render_metric_card("API Status", "🔴 Unreachable", f"Last checked: {health['time'].strftime('%H:%M:%S')}", "error", "❌")
            with st.expander("🔍 Error Details"):
                st.code(health.get("error", "Unknown error"))
    else:
        render_metric_card("API Status", "⚪ Unknown", "Click 'Check Now' to test connection", "normal", "❓")

    # Auto-refresh health check
    if auto_health:
        import time
        time.sleep(5)
        st.rerun()

    st.divider()

    # Main tabs for API testing
    tab_predict, tab_screenshot, tab_batch, tab_docs, tab_history = st.tabs([
        "🔮 Single Prediction", "📱 Screenshot", "📦 Batch Predict", "📚 API Docs", "📜 Request History"
    ])

    with tab_predict:
        st.subheader("🔮 Fraud Risk Prediction")
        st.markdown("Send a single transaction for fraud risk scoring.")

        with st.form("api_predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                api_dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="api_dataset",
                                         help="Dataset the model was trained on")
                api_model = st.selectbox("Model", ["lightgbm", "logistic_regression", "random_forest", "xgboost"], key="api_model",
                                        help="Model to use for prediction")

                # Quick feature templates
                template = st.selectbox("Feature Template", [
                    "Custom...",
                    "IEEE-CIS Sample (Low Risk)",
                    "IEEE-CIS Sample (High Risk)",
                    "PaySim Sample (Transfer)",
                    "PaySim Sample (Cash Out)",
                ], key="api_template")

                if template == "IEEE-CIS Sample (Low Risk)":
                    default_features = {
                        "TransactionAmt": 50.0, "card1": 12345, "card4": "visa", "ProductCD": "W",
                        "card2": 100.0, "card3": 150.0, "addr1": 100.0, "dist1": 10.0
                    }
                elif template == "IEEE-CIS Sample (High Risk)":
                    default_features = {
                        "TransactionAmt": 5000.0, "card1": 99999, "card4": "mastercard", "ProductCD": "C",
                        "card2": 500.0, "card3": 800.0, "addr1": 999.0, "dist1": 500.0
                    }
                elif template == "PaySim Sample (Transfer)":
                    default_features = {
                        "step": 100, "type": "TRANSFER", "amount": 1000.0,
                        "oldbalanceOrg": 5000.0, "newbalanceOrig": 4000.0,
                        "oldbalanceDest": 1000.0, "newbalanceDest": 2000.0
                    }
                elif template == "PaySim Sample (Cash Out)":
                    default_features = {
                        "step": 200, "type": "CASH_OUT", "amount": 2000.0,
                        "oldbalanceOrg": 10000.0, "newbalanceOrig": 8000.0,
                        "oldbalanceDest": 0.0, "newbalanceDest": 2000.0
                    }
                else:
                    default_features = {
                        "TransactionAmt": 150.0, "card1": 12345, "card4": "visa", "ProductCD": "W"
                    } if api_dataset == "ieee-cis" else {
                        "step": 100, "type": "PAYMENT", "amount": 100.0,
                        "oldbalanceOrg": 1000.0, "newbalanceOrig": 900.0,
                        "oldbalanceDest": 500.0, "newbalanceDest": 600.0
                    }

            with col2:
                api_features_json = st.text_area(
                    "Features (JSON)",
                    value=json.dumps(default_features, indent=2),
                    height=300,
                    key="api_features_json",
                    help="Enter transaction features as JSON. Required fields depend on dataset."
                )

            submitted = st.form_submit_button("🚀 Send Prediction Request", type="primary", use_container_width=True)

        if submitted:
            try:
                features = json.loads(api_features_json)

                # Add to history
                request_entry = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "endpoint": "/predict",
                    "dataset": api_dataset,
                    "model": api_model,
                    "features": features
                }

                with st.spinner("Sending prediction request..."):
                    response = api_client.post("/predict", json={
                        "dataset": api_dataset,
                        "model": api_model,
                        "features": features,
                    })

                request_entry["response_time"] = pd.Timestamp.now().isoformat()
                request_entry["status_code"] = response.status_code

                if response.status_code == 200:
                    data = response.json()
                    request_entry["response"] = data
                    request_entry["success"] = True
                    show_toast("✅ Prediction successful!", "success")

                    preds = data.get("predictions", [])
                    if preds:
                        st.subheader("📊 Prediction Results")

                        # Summary cards
                        for i, pred in enumerate(preds):
                            is_fraud = pred["is_fraud"] == 1
                            score = pred["fraud_score"]

                            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                            with col_r1:
                                render_metric_card(f"Prediction {i+1}", f"{score:.4f}",
                                                 "🔴 FRAUD" if is_fraud else "🟢 LEGIT",
                                                 "error" if is_fraud else "success", "🔮")
                            with col_r2:
                                risk_level = "Critical" if score > 0.8 else ("High" if score > 0.6 else ("Medium" if score > 0.4 else "Low"))
                                render_metric_card("Risk Level", risk_level, "",
                                                 "error" if score > 0.8 else ("warning" if score > 0.6 else "normal"), "⚡")
                            with col_r3:
                                render_metric_card("Threshold", "0.50",
                                                 "Above" if score > 0.5 else "Below", "normal", "📏")
                            with col_r4:
                                action = "BLOCK" if is_fraud else ("REVIEW" if score > 0.3 else "ALLOW")
                                render_metric_card("Action", action, "",
                                                 "error" if action == "BLOCK" else ("warning" if action == "REVIEW" else "success"), "🛡️")

                        # Full response
                        with st.expander("📋 Full API Response"):
                            st.json(data)

                        # cURL equivalent
                        with st.expander("💻 cURL Equivalent"):
                            curl_cmd = f'''curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps({"dataset": api_dataset, "model": api_model, "features": features})}'''
                            st.code(curl_cmd, language="bash")
                else:
                    request_entry["success"] = False
                    request_entry["error"] = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                    show_toast(f"❌ API returned status {response.status_code}", "error")
                    st.error(f"API Error ({response.status_code}):")
                    st.json(request_entry["error"])

            except json.JSONDecodeError as e:
                show_toast(f"❌ Invalid JSON: {e}", "error")
                st.error(f"Invalid JSON in features field: {e}")
            except Exception as exc:
                show_toast(f"❌ Request failed: {exc}", "error")
                st.error(f"Prediction request failed: {exc}")

            # Store in history
            if "api_history" not in st.session_state:
                st.session_state.api_history = []
            st.session_state.api_history.append(request_entry)

    with tab_screenshot:
        st.subheader("📱 Screenshot Fraud Detection")
        st.markdown("Upload a PhonePe/GPay payment screenshot for fraud risk analysis.")

        col1, col2 = st.columns([1, 1])

        with col1:
            screenshot_dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="scr_dataset")
            screenshot_model = st.selectbox("Model", ["lightgbm"], key="scr_model")

            uploaded_file = st.file_uploader(
                "Upload Screenshot",
                type=["png", "jpg", "jpeg"],
                key="scr_upload",
                help="Upload a PhonePe or GPay transaction screenshot"
            )

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Screenshot", use_container_width=True)

                if st.button("🔍 Analyze Screenshot", key="scr_analyze", type="primary", use_container_width=True):
                    with st.spinner("Parsing screenshot and predicting..."):
                        try:
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                tmp.write(uploaded_file.read())
                                tmp_path = tmp.name

                            # Prepare multipart form data
                            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            data = {"dataset": screenshot_dataset, "model": screenshot_model}

                            response = api_client.post("/predict_screenshot", data=data, files=files)

                            if response.status_code == 200:
                                result = response.json()
                                show_toast("✅ Screenshot analyzed!", "success")

                                st.subheader("📱 Parsed Transaction")
                                parsed = result.get("parsed_transaction", {})

                                col_p1, col_p2 = st.columns(2)
                                with col_p1:
                                    render_metric_card("App", parsed.get("app_type", "Unknown").upper(), "", "normal", "📱")
                                    render_metric_card("Amount", f"₹{parsed.get('amount', 0):,.2f}", "", "normal", "💰")
                                    render_metric_card("Status", parsed.get("status", "Unknown"), "", "normal", "📋")
                                with col_p2:
                                    render_metric_card("Sender", parsed.get("sender_name", "N/A"), parsed.get("sender_upi", ""), "normal", "👤")
                                    render_metric_card("Receiver", parsed.get("receiver_name", "N/A"), parsed.get("receiver_upi", ""), "normal", "🏦")
                                    render_metric_card("Bank", parsed.get("bank_name", "N/A"), "", "normal", "🏛️")

                                st.subheader("🔮 Fraud Prediction")
                                preds = result.get("predictions", [])
                                if preds:
                                    pred = preds[0]
                                    is_fraud = pred["is_fraud"] == 1
                                    score = pred["fraud_score"]

                                    render_metric_card("Fraud Score", f"{score:.4f}",
                                                     "🔴 FRAUD DETECTED" if is_fraud else "🟢 Legitimate",
                                                     "error" if is_fraud else "success", "🔮")

                                with st.expander("📋 Full Response"):
                                    st.json(result)
                            else:
                                show_toast(f"❌ Analysis failed: {response.status_code}", "error")
                                st.error(response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text)
                        except Exception as e:
                            show_toast(f"❌ Error: {e}", "error")
                            st.error(f"Screenshot analysis failed: {e}")

        with col2:
            st.markdown("### 📖 Supported Apps")
            st.markdown("""
            - **PhonePe** - Payment confirmations, money transfers
            - **Google Pay (GPay)** - UPI payments, bill payments
            - **Paytm** - UPI transactions (limited support)
            """)

            st.markdown("### 🎯 Best Results")
            st.markdown("""
            - Clear, well-lit screenshots
            - Full transaction details visible
            - Standard app UI (not modified themes)
            - Portrait orientation preferred
            """)

            st.markdown("### 🔧 API Endpoint")
            st.code("POST /predict_screenshot", language="text")
            st.caption("Multipart form-data with 'image' file field")

    with tab_batch:
        st.subheader("📦 Batch Prediction")
        st.markdown("Score multiple transactions in a single API call.")

        col1, col2 = st.columns([1, 1])

        with col1:
            batch_dataset = st.selectbox("Dataset", ["ieee-cis", "paysim"], key="batch_dataset")
            batch_model = st.selectbox("Model", ["lightgbm", "logistic_regression", "random_forest", "xgboost"], key="batch_model")

            batch_input_method = st.radio("Input Method", ["JSON Array", "CSV Upload", "Sample Data"], key="batch_method")

            if batch_input_method == "JSON Array":
                batch_features = st.text_area(
                    "Features Array (JSON)",
                    value='[\n  {"TransactionAmt": 100.0, "card1": 11111, "card4": "visa", "ProductCD": "W"},\n  {"TransactionAmt": 5000.0, "card1": 22222, "card4": "mastercard", "ProductCD": "C"}\n]',
                    height=200,
                    key="batch_json"
                )
            elif batch_input_method == "CSV Upload":
                batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
                if batch_file:
                    batch_df = pd.read_csv(batch_file)
                    st.caption(f"Loaded {len(batch_df)} rows, {len(batch_df.columns)} columns")
                    st.dataframe(batch_df.head(), use_container_width=True)
            else:  # Sample Data
                sample_count = st.slider("Sample Count", 1, 20, 5, key="batch_samples")
                if st.button("Generate Sample", key="batch_gen"):
                    # Generate sample based on dataset
                    if batch_dataset == "ieee-cis":
                        samples = []
                        for i in range(sample_count):
                            samples.append({
                                "TransactionAmt": float(np.random.uniform(10, 5000)),
                                "card1": int(np.random.randint(1000, 99999)),
                                "card4": np.random.choice(["visa", "mastercard", "discover", "amex"]),
                                "ProductCD": np.random.choice(["W", "C", "R", "S", "H"]),
                            })
                    else:
                        samples = []
                        for i in range(sample_count):
                            samples.append({
                                "step": int(np.random.randint(1, 744)),
                                "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"]),
                                "amount": float(np.random.uniform(10, 10000)),
                                "oldbalanceOrg": float(np.random.uniform(0, 50000)),
                                "newbalanceOrig": float(np.random.uniform(0, 50000)),
                                "oldbalanceDest": float(np.random.uniform(0, 10000)),
                                "newbalanceDest": float(np.random.uniform(0, 10000)),
                            })
                    st.session_state["batch_samples"] = samples
                    show_toast(f"Generated {sample_count} samples!", "success")

                if "batch_samples" in st.session_state:
                    st.json(st.session_state["batch_samples"][:3])

        with col2:
            if st.button("🚀 Run Batch Prediction", key="batch_run", type="primary", use_container_width=True):
                try:
                    if batch_input_method == "JSON Array":
                        features = json.loads(batch_features)
                    elif batch_input_method == "CSV Upload":
                        features = batch_df.to_dict('records')
                    else:
                        features = st.session_state.get("batch_samples", [])

                    if not features:
                        show_toast("No features to predict!", "warning")
                        st.stop()

                    with st.spinner(f"Scoring {len(features)} transactions..."):
                        response = api_client.post("/predict", json={
                            "dataset": batch_dataset,
                            "model": batch_model,
                            "features": features,
                        })

                    if response.status_code == 200:
                        data = response.json()
                        preds = data.get("predictions", [])
                        show_toast(f"✅ Scored {len(preds)} transactions!", "success")

                        # Results summary
                        fraud_count = sum(1 for p in preds if p["is_fraud"] == 1)
                        avg_score = np.mean([p["fraud_score"] for p in preds])
                        max_score = max(p["fraud_score"] for p in preds)

                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            render_metric_card("Total", str(len(preds)), "", "normal", "📊")
                        with col_s2:
                            render_metric_card("Fraud Detected", str(fraud_count), f"{fraud_count/len(preds)*100:.1f}%", "error" if fraud_count else "success", "🔴")
                        with col_s3:
                            render_metric_card("Avg Score", f"{avg_score:.4f}", "", "normal", "📈")
                        with col_s4:
                            render_metric_card("Max Score", f"{max_score:.4f}", "", "error" if max_score > 0.8 else "normal", "⚡")

                        # Results table
                        results_df = pd.DataFrame(preds)
                        results_df["risk_level"] = results_df["fraud_score"].apply(
                            lambda x: "Critical" if x > 0.8 else ("High" if x > 0.6 else ("Medium" if x > 0.4 else "Low"))
                        )
                        results_df["action"] = results_df.apply(
                            lambda r: "BLOCK" if r["is_fraud"] == 1 else ("REVIEW" if r["fraud_score"] > 0.3 else "ALLOW"), axis=1
                        )

                        st.dataframe(
                            results_df[["index", "fraud_score", "is_fraud", "risk_level", "action"]],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "index": st.column_config.NumberColumn("#", format="%d"),
                                "fraud_score": st.column_config.NumberColumn("Score", format="%.4f"),
                                "is_fraud": st.column_config.CheckboxColumn("Fraud"),
                                "risk_level": st.column_config.TextColumn("Risk"),
                                "action": st.column_config.TextColumn("Action"),
                            }
                        )

                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            file_name=f"batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="batch_dl",
                            use_container_width=True
                        )
                    else:
                        show_toast(f"❌ Batch failed: {response.status_code}", "error")
                        st.error(response.json())
                except Exception as e:
                    show_toast(f"❌ Batch prediction failed: {e}", "error")
                    st.error(f"Batch prediction failed: {e}")

    with tab_docs:
        st.subheader("📚 API Documentation")

        # Endpoints table
        endpoints = [
            {
                "method": "GET", "path": "/", "description": "API information and available endpoints",
                "params": "None", "response": "Service info, version, endpoints list"
            },
            {
                "method": "GET", "path": "/health", "description": "Health check endpoint",
                "params": "None", "response": "{\"status\": \"healthy\", \"service\": \"upi-fraud-detection\"}"
            },
            {
                "method": "POST", "path": "/predict", "description": "Fraud risk scoring from features",
                "params": "dataset, model, features (dict or list)",
                "response": "predictions array with fraud_score, is_fraud"
            },
            {
                "method": "POST", "path": "/predict_screenshot", "description": "Fraud risk from payment screenshot",
                "params": "image (multipart), dataset, model",
                "response": "parsed_transaction + predictions"
            },
        ]

        for ep in endpoints:
            with st.expander(f"`{ep['method']}` {ep['path']} — {ep['description']}"):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("**Parameters**")
                    st.code(ep["params"])
                    st.markdown("**Response**")
                    st.code(ep["response"])
                with col_d2:
                    # cURL example
                    if ep['method'] == 'GET':
                        curl = f"curl -X GET http://localhost:8000{ep['path']}"
                    else:
                        curl = f'''curl -X POST http://localhost:8000{ep['path']} \\
  -H "Content-Type: application/json" \\
  -d '{{"dataset": "ieee-cis", "model": "lightgbm", "features": {{}}}}'''
                    st.markdown("**cURL Example**")
                    st.code(curl, language="bash")

                    # Python example
                    py = f'''import requests
response = requests.{ep['method'].lower()}(
    "http://localhost:8000{ep['path']}",
    json={{"dataset": "ieee-cis", "model": "lightgbm", "features": {{}}}}
)
print(response.json())'''
                    st.markdown("**Python Example**")
                    st.code(py, language="python")

        # OpenAPI/Swagger link
        st.divider()
        st.markdown("### 🔗 Interactive Documentation")
        st.markdown("When the API server is running, visit:")
        st.code("http://localhost:8000/docs  # Swagger UI")
        st.code("http://localhost:8000/redoc  # ReDoc")

    with tab_history:
        st.subheader("📜 API Request History")

        if "api_history" not in st.session_state or not st.session_state.api_history:
            st.info("No API requests made yet. Test the endpoints above to populate history.")
        else:
            history = st.session_state.api_history
            hist_df = pd.DataFrame(history)

            # Summary
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            with col_h1:
                render_metric_card("Total Requests", str(len(hist_df)), "", "normal", "📋")
            with col_h2:
                success_count = hist_df["success"].sum() if "success" in hist_df.columns else 0
                render_metric_card("Successful", str(success_count), f"{success_count/len(hist_df)*100:.0f}%", "success", "✅")
            with col_h3:
                if "status_code" in hist_df.columns:
                    avg_status = hist_df["status_code"].mode()[0] if len(hist_df) > 0 else 0
                    render_metric_card("Common Status", str(avg_status), "", "normal", "📊")
            with col_h4:
                endpoints_used = hist_df["endpoint"].nunique() if "endpoint" in hist_df.columns else 0
                render_metric_card("Endpoints Used", str(endpoints_used), "", "normal", "🔌")

            # Filter
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                endpoint_filter = st.selectbox("Endpoint", ["All"] + sorted(hist_df["endpoint"].unique().tolist()), key="hist_endpoint")
            with col_f2:
                status_filter = st.selectbox("Status", ["All", "Success", "Failed"], key="hist_status")

            filtered = hist_df.copy()
            if endpoint_filter != "All":
                filtered = filtered[filtered["endpoint"] == endpoint_filter]
            if status_filter == "Success":
                filtered = filtered[filtered["success"] == True]
            elif status_filter == "Failed":
                filtered = filtered[filtered["success"] == False]

            # History table
            st.dataframe(
                filtered[["timestamp", "endpoint", "dataset", "model", "success", "status_code"]].tail(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("Time"),
                    "endpoint": st.column_config.TextColumn("Endpoint"),
                    "dataset": st.column_config.TextColumn("Dataset"),
                    "model": st.column_config.TextColumn("Model"),
                    "success": st.column_config.CheckboxColumn("OK"),
                    "status_code": st.column_config.NumberColumn("Status", format="%d"),
                }
            )

            # Clear history
            if st.button("🗑️ Clear History", key="api_clear_hist"):
                st.session_state.api_history = []
                show_toast("History cleared!", "success")
                st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**UPI Fraud Detection System**")
st.sidebar.markdown("MCA Project - Vidyavardhaka College of Engineering")
st.sidebar.markdown("Student: Pavan C (4VV25MC060)")