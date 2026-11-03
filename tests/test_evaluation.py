"""Unit tests for evaluation module — including the single-class confusion matrix fix."""

import numpy as np
import pytest

from src.evaluation import compute_operational_metrics, find_optimal_threshold


def test_compute_operational_metrics_single_class_true_all_pred_zero():
    """Confusion matrix must not crash when a window is fraud-free (all y_pred=0)."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.05, 0.15, 0.3])
    metrics = compute_operational_metrics(y_true, y_score, total_transactions=10000)
    assert metrics["true_negatives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_compute_operational_metrics_single_class_true_all_pred_one():
    """All fraud window, model predicts all 1."""
    y_true = np.array([1, 1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.85, 0.7, 0.95])
    metrics = compute_operational_metrics(y_true, y_score)
    assert metrics["true_negatives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_compute_operational_metrics_mixed():
    """Normal mixed-class window."""
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.05, 0.1, 0.2])
    metrics = compute_operational_metrics(y_true, y_score)
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
        == len(y_true)
    )


def test_find_optimal_threshold_f1():
    """find_optimal_threshold returns a valid threshold in [0, 1]."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.1, 1000)
    y_score = rng.beta(1, 50, 1000)
    y_score[y_true == 1] = rng.beta(3, 10, int(y_true.sum()))
    threshold, metrics = find_optimal_threshold(y_true, y_score, metric="f1")
    assert 0.0 <= threshold <= 1.0
    assert "f1" in metrics
