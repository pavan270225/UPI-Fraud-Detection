"""
Evaluation module for UPI Fraud Detection pipeline.

Computes operational metrics: PR-AUC, Precision@K, Recall@K, FP/100k, latency, retraining frequency.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
import warnings
warnings.filterwarnings('ignore')


class EvaluationError(Exception):
    """Raised when evaluation fails."""
    pass


def compute_operational_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: List[int] = [50, 100, 200, 500, 1000],
    alert_budget: int = 100,
    total_transactions: Optional[int] = None,
) -> Dict[str, float]:
    """Compute operational fraud detection metrics."""
    metrics = {}

    metrics['pr_auc'] = average_precision_score(y_true, y_score)
    metrics['roc_auc'] = roc_auc_score(y_true, y_score)

    y_pred = (y_score >= 0.5).astype(int)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    n_samples = len(y_true)
    for k in k_values:
        if k <= n_samples:
            top_k_idx = np.argsort(y_score)[-k:]
            metrics[f'precision_at_{k}'] = precision_score(y_true[top_k_idx], np.ones(k), zero_division=0)
            metrics[f'recall_at_{k}'] = recall_score(y_true[top_k_idx], np.ones(k), zero_division=0)

    if total_transactions and total_transactions > 0:
        if alert_budget <= n_samples:
            threshold = np.sort(y_score)[-alert_budget]
            y_pred_budget = (y_score >= threshold).astype(int)
            fp = np.sum((y_pred_budget == 1) & (y_true == 0))
            metrics[f'fp_per_100k_at_budget_{alert_budget}'] = (fp / total_transactions) * 100_000

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['fraud_rate'] = y_true.mean()
    metrics['alert_rate'] = y_pred.mean()

    return metrics


def compute_retraining_metrics(
    retraining_history: pd.DataFrame,
    total_days: int,
) -> Dict[str, float]:
    """Compute retraining operational metrics."""
    if retraining_history.empty:
        return {
            'retrain_count': 0,
            'retrain_frequency_per_month': 0,
            'avg_model_age_days': 0,
            'max_model_age_days': 0,
        }

    retrain_count = len(retraining_history)
    retrain_frequency = (retrain_count / total_days) * 30
    model_ages = retraining_history['model_age_days'].values
    avg_age = model_ages.mean() if len(model_ages) > 0 else 0
    max_age = model_ages.max() if len(model_ages) > 0 else 0
    trigger_counts = retraining_history['trigger'].value_counts().to_dict()

    return {
        'retrain_count': int(retrain_count),
        'retrain_frequency_per_month': float(retrain_frequency),
        'avg_model_age_days': float(avg_age),
        'max_model_age_days': float(max_age),
        'trigger_breakdown': trigger_counts,
    }


def compare_strategies_statistically(
    results_df: pd.DataFrame,
    metric: str = 'pr_auc',
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Statistical comparison of retraining strategies."""
    strategies = results_df['strategy'].unique()
    comparisons = {}

    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            if 'window_start' in results_df.columns:
                merged = results_df[results_df['strategy'].isin([s1, s2])].pivot(
                    index='window_start', columns='strategy', values=metric
                ).dropna()

                if len(merged) >= 2:
                    stat, p_val = stats.wilcoxon(merged[s1], merged[s2], alternative='two-sided')
                    comparisons[f'{s1}_vs_{s2}'] = {
                        'statistic': float(stat),
                        'p_value': float(p_val),
                        'significant': p_val < alpha,
                        'mean_diff': float(merged[s1].mean() - merged[s2].mean()),
                        's1_mean': float(merged[s1].mean()),
                        's2_mean': float(merged[s2].mean()),
                    }

    return comparisons


def compute_latency_metrics(latencies: List[float]) -> Dict[str, float]:
    """Compute latency statistics."""
    if not latencies:
        return {}
    arr = np.array(latencies)
    return {
        'latency_mean_ms': float(np.mean(arr) * 1000),
        'latency_p50_ms': float(np.percentile(arr, 50) * 1000),
        'latency_p95_ms': float(np.percentile(arr, 95) * 1000),
        'latency_p99_ms': float(np.percentile(arr, 99) * 1000),
        'latency_max_ms': float(np.max(arr) * 1000),
    }


def generate_evaluation_report(
    model_name: str,
    dataset: str,
    operational_metrics: Dict,
    retraining_metrics: Dict,
    cv_metrics: Optional[Dict] = None,
    statistical_comparison: Optional[Dict] = None,
) -> str:
    """Generate human-readable evaluation report."""
    lines = [
        "=" * 60,
        f"EVALUATION REPORT: {model_name} on {dataset}",
        "=" * 60,
        "",
        "OPERATIONAL METRICS:",
        f"  PR-AUC: {operational_metrics.get('pr_auc', 0):.4f}",
        f"  ROC-AUC: {operational_metrics.get('roc_auc', 0):.4f}",
        f"  Precision@100: {operational_metrics.get('precision_at_100', 0):.4f}",
        f"  Recall@100: {operational_metrics.get('recall_at_100', 0):.4f}",
        f"  FP/100k (budget=100): {operational_metrics.get('fp_per_100k_at_budget_100', 0):.2f}",
        f"  Fraud Rate: {operational_metrics.get('fraud_rate', 0):.4%}",
        f"  Alert Rate: {operational_metrics.get('alert_rate', 0):.4%}",
        "",
        "RETRAINING METRICS:",
        f"  Retrain Count: {retraining_metrics.get('retrain_count', 0)}",
        f"  Frequency (per month): {retraining_metrics.get('retrain_frequency_per_month', 0):.2f}",
        f"  Avg Model Age (days): {retraining_metrics.get('avg_model_age_days', 0):.1f}",
        f"  Max Model Age (days): {retraining_metrics.get('max_model_age_days', 0):.1f}",
    ]

    if cv_metrics:
        lines.extend([
            "",
            "CROSS-VALIDATION:",
            f"  PR-AUC Mean: {cv_metrics.get('pr_auc_mean', 0):.4f} +/- {cv_metrics.get('pr_auc_std', 0):.4f}",
            f"  P@100 Mean: {cv_metrics.get('p100_mean', 0):.4f}",
        ])

    if statistical_comparison:
        lines.extend(["", "STATISTICAL COMPARISONS:"])
        for comp, result in statistical_comparison.items():
            sig = "PASS" if result.get('significant') else "FAIL"
            lines.append(f"  {comp}: mean_diff={result.get('mean_diff', 0):.4f}, p={result.get('p_value', 1):.4f} {sig}")

    return "\n".join(lines)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = 'f1',
) -> Tuple[float, Dict]:
    """Find optimal decision threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_metrics = {
            'threshold': float(best_threshold),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx]),
            'f1': float(f1_scores[best_idx]),
        }
    elif metric == 'precision':
        min_recall = 0.5
        valid = recall[:-1] >= min_recall
        if np.any(valid):
            best_idx = np.argmax(precision[:-1][valid])
            best_idx = np.where(valid)[0][best_idx]
        else:
            best_idx = np.argmax(precision[:-1])
        best_threshold = thresholds[best_idx]
        best_metrics = {
            'threshold': float(best_threshold),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx]),
        }
    else:
        best_threshold = 0.5
        best_metrics = {'threshold': 0.5}

    return float(best_threshold), best_metrics


if __name__ == '__main__':
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.01, 10000)
    y_score = np.random.beta(1, 50, 10000)
    y_score[y_true == 1] = np.random.beta(3, 10, y_true.sum())

    metrics = compute_operational_metrics(y_true, y_score, total_transactions=10000)
    print("Operational metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")