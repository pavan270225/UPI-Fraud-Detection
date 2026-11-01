# UPI Fraud Detection System

**A Drift-Aware and Explainable Machine Learning Framework for Fraud Detection in UPI Digital Payments under Delayed Labels**

## 📋 Project Overview

This project implements a comprehensive fraud detection pipeline for UPI (Unified Payments Interface) digital payments, addressing key challenges:

- **Concept Drift**: Models degrade over time as fraud patterns evolve
- **Delayed Labels**: Fraud labels arrive days/weeks after transactions
- **Explainability**: Regulatory requirements demand per-transaction explanations
- **Operational Metrics**: Business-relevant metrics (FP/100k, P@K) over academic AUC

### Key Features

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data** | IEEE-CIS, PaySim | Time-ordered proxy datasets |
| **Models** | LR, RF, XGBoost, LightGBM | Rolling time-window CV |
| **Drift Detection** | PSI, KS tests | Label-free feature & score monitoring |
| **Explainability** | TreeSHAP | Per-instance + global importance |
| **Retraining** | Static / Scheduled / Drift-Triggered | Strategy comparison |
| **Dashboard** | Streamlit | Interactive visualization |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- 8GB+ RAM recommended

### 1. Clone & Setup

```bash
cd UPI_Fraud_Detection_Project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets

**Required** (Kaggle login needed):
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) → `data/ieee-cis/train_transaction.csv`
- [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) → `data/paysim/paysim.csv`

### 3. Run Pipeline (Notebooks)

```bash
jupyter notebook notebooks/
```

Execute in order:
1. `01_eda.ipynb` - EDA, time-ordered splits, feature engineering
2. `02_baselines.ipynb` - Train 4 models with rolling CV
3. `03_delayed_labels.ipynb` - Simulate 1/7/30-day label delays
4. `04_drift_monitoring.ipynb` - PSI/KS drift detection
5. `05_shap_explanations.ipynb` - SHAP explanations & stability
6. `06_retraining_strategies.ipynb` - Compare retraining strategies
7. `07_evaluation.ipynb` - Final operational evaluation

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t upi-fraud .

# Run (mount data directory)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  upi-fraud
```

---

## 📁 Project Structure

```
UPI_Fraud_Detection_Project/
├── data/
│   ├── ieee-cis/           # IEEE-CIS dataset (user downloads)
│   └── paysim/             # PaySim dataset (user downloads)
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_delayed_labels.ipynb
│   ├── 04_drift_monitoring.ipynb
│   ├── 05_shap_explanations.ipynb
│   ├── 06_retraining_strategies.ipynb
│   └── 07_evaluation.ipynb
├── reports/                # Generated metrics & reports
├── src/
│   ├── data_prep.py        # Data loading, FE, splits
│   ├── models.py           # Model factory, training, CV
│   ├── drift_detection.py  # PSI/KS, DriftMonitor
│   ├── shap_explainer.py   # TreeSHAP wrapper
│   ├── retraining.py       # 3 strategies + orchestrator
│   └── evaluation.py       # Operational metrics
├── tests/                  # Unit tests
├── Dockerfile
├── requirements.txt
└── PRESENTATION_AND_PROJECT_PLAN.md
```

---

## 📊 Target Metrics

| Metric | Target | Status |
|--------|--------|--------|
| PR-AUC | > 0.85 | 🎯 |
| Precision@100 | > 0.70 | 🎯 |
| FP/100k (budget=100) | < 50 | 🎯 |
| Drift detection latency | < 5 min | 🎯 |
| Retrain reduction (drift vs scheduled) | > 30% | 🎯 |
| SHAP explanation latency | < 200ms | 🎯 |

---

## 🔬 Methodology

### 1. Time-Ordered Splits
- **Train**: Days 0-49 (70%)
- **Validation**: Days 50-59 (15%)
- **Test**: Days 60+ (15%)
- **No leakage**: Strict temporal ordering

### 2. Rolling Cross-Validation
- 5 folds, expanding window
- Each fold: train on past, validate on future
- Mimics production deployment

### 3. Delayed Labels Simulation
- Oracle: 0-day delay (upper bound)
- Realistic: 1, 7, 30-day delays
- Rolling retraining with delayed labels

### 4. Drift Monitoring (Label-Free)
- **PSI** (Population Stability Index): Feature distribution shift
- **KS** (Kolmogorov-Smirnov): Score distribution shift
- Thresholds: PSI > 0.2, KS > 0.05
- Rolling 7-day windows

### 5. SHAP Explainability
- TreeSHAP for tree-based models
- Per-alert waterfall plots
- Global importance tracking
- Stability via coefficient of variation

### 6. Retraining Strategies
| Strategy | Trigger | Expected Retrains/Month |
|----------|---------|------------------------|
| Static | Never | 0 |
| Scheduled | Every 7 days | ~4 |
| Drift-Triggered | PSI/KS alert | ~2-3 (30% fewer) |

---

## 📈 Results

Results are generated in `reports/` after running notebooks:
- `ieee_test_metrics.csv` / `paysim_test_metrics.csv` - Baseline performance
- `drift_alerts_log.csv` - PSI/KS alerts
- `shap_global_importance.csv` - Feature importance
- `shap_stability.csv` - Importance stability over time
- `retraining_strategy_comparison.csv` - Strategy comparison
- `delay_impact_summary.csv` - Delayed labels impact
- `final_evaluation_report.txt` - Consolidated report

---

## 🎓 Academic Context

**Student**: Pavan C (USN: 4VV25MC060)  
**College**: Vidyavardhaka College of Engineering  
**Course**: MCA, Semester 3  
**Guide**: Dhanvi L  

**Presentation**: `Project_Presentation_Pavan_C_UPI_Fraud_Detection.pptx` (16 slides)

---

## 📚 References

Key papers implemented:
1. **Concept Drift** - Gama et al. (2014), "A survey on concept drift adaptation"
2. **Delayed Labels** - Xu et al. (2024), "Delayed feedback in fraud detection"
3. **PSI/KS Drift** - Lu et al. (2018), "Detecting concept drift with PSI"
4. **SHAP** - Lundberg & Lee (2017), "Unified approach to interpreting model predictions"
5. **Recent 2025-26 papers** on streaming fraud detection & LLM tabular learning

---

## 📝 License

Academic project - Vidyavardhaka College of Engineering

---

## 🤝 Contributing

This is an academic project. For questions or collaboration:
- Open an issue
- Contact: Pavan C (4VV25MC060)