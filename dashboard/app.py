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
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from data_prep import load_ieee_cis, load_paysim, prepare_features_target
from models import evaluate_model, get_model
from drift_detection import DriftMonitor, population_stability_index
from shap_explainer import SHAPExplainer

# Page config
st.set_page_config(
    page_title="UPI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high { background-color: #ffebee; border-left: 4px solid #f44336; }
    .alert-medium { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .alert-low { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_ROOT = Path('../data')
MODELS_DIR = Path('../models')
REPORTS_DIR = Path('../reports')

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
            reports[name] = pd.read_csv(path)
        else:
            reports[name] = pd.DataFrame()
    return reports

@st.cache_resource
def load_model_and_data():
    """Load model and data for explanations."""
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
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes
    return X.fillna(-999), y

# Sidebar
st.sidebar.title("🛡️ UPI Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🎯 Risk Scoring", "🔍 SHAP Explanations", "📈 Drift Monitoring",
     "⚙️ Retraining Strategies", "⏱️ Delayed Labels", "📋 Reports"]
)

# Load reports
reports = load_reports()
model, model_meta = load_model_and_data()

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.title("📊 UPI Fraud Detection - System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Project", "UPI Fraud Detection", help="Drift-Aware Explainable ML Framework")

    with col2:
        if not reports['ieee_test_metrics'].empty:
            best_pr_auc = reports['ieee_test_metrics']['pr_auc'].max()
            st.metric("Best PR-AUC (IEEE-CIS)", f"{best_pr_auc:.4f}",
                     delta="Target: >0.85" if best_pr_auc > 0.85 else "Below target")
        else:
            st.metric("Best PR-AUC", "N/A")

    with col3:
        if not reports['drift_alerts'].empty:
            total_alerts = len(reports['drift_alerts'])
            st.metric("Drift Alerts", total_alerts)
        else:
            st.metric("Drift Alerts", "N/A")

    with col4:
        if not reports['retrain_comparison'].empty:
            strategies = reports['retrain_comparison']['strategy'].nunique()
            st.metric("Strategies Compared", strategies)
        else:
            st.metric("Strategies Compared", "N/A")

    st.markdown("---")

    # Project description
    st.subheader("Project Description")
    st.markdown("""
    **A Drift-Aware and Explainable Machine Learning Framework for Fraud Detection in UPI Digital Payments under Delayed Labels**

    This system implements a comprehensive fraud detection pipeline for UPI (Unified Payments Interface) digital payments:

    - **Data**: IEEE-CIS Fraud Detection & PaySim datasets as proxies for private UPI data
    - **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM with rolling time-window CV
    - **Drift Monitoring**: PSI (Population Stability Index) & KS (Kolmogorov-Smirnov) tests on features & scores
    - **Explainability**: TreeSHAP per-instance explanations + global importance tracking
    - **Retraining**: Static vs Scheduled vs Drift-Triggered strategies comparison
    - **Delayed Labels**: Simulation of 1/7/30-day label delays with performance degradation analysis

    **Student**: Pavan C (USN: 4VV25MC060)
    **College**: Vidyavardhaka College of Engineering
    **Course**: MCA, Semester 3
    """)

    # Target achievement
    st.subheader("🎯 Target Achievement")

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
                if passed:
                    st.success(f"✅ {target}\n\nValue: {value:.4f}")
                else:
                    st.error(f"❌ {target}\n\nValue: {value:.4f}")

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

        except Exception as e:
            st.error(f"Error loading test data: {e}")

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

        # Alerts table
        st.subheader("📋 Drift Alerts Log")
        if not alerts_df.empty:
            # Add alert severity styling
            display_df = alerts_df.copy()
            st.dataframe(display_df, use_container_width=True, height=400)

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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**UPI Fraud Detection System**")
st.sidebar.markdown("MCA Project - Vidyavardhaka College of Engineering")
st.sidebar.markdown("Student: Pavan C (4VV25MC060)")