"""Unit tests for drift_detection module — PSI and KS."""

import numpy as np
import pytest

from src.drift_detection import kolmogorov_smirnov_test, population_stability_index


def test_psi_identical_distributions_is_zero():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    cur = ref.copy()  # identical -> PSI should be ~0
    psi, _ = population_stability_index(ref, cur)
    assert psi < 0.05  # essentially zero


def test_psi_shifted_distribution_detected():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(1.0, 1.0, 1000)  # shifted mean
    psi, _ = population_stability_index(ref, cur)
    assert psi > 0.1  # meaningful shift should be detected


def test_psi_empty_array_handled():
    psi, details = population_stability_index(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(psi)
    assert "error" in details


def test_ks_identical_distributions_high_pvalue():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 500)
    cur = ref.copy()
    ks_stat, p_val, _ = kolmogorov_smirnov_test(ref, cur)
    assert p_val > 0.05  # identical distributions -> not significant


def test_ks_shifted_low_pvalue():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(1.0, 1.0, 1000)
    ks_stat, p_val, _ = kolmogorov_smirnov_test(ref, cur)
    assert p_val < 0.05  # shifted -> significant


def test_summarize_drift_alerts():
    """Alert aggregation should count features and types correctly."""
    results = [
        {
            "alerts": [
                {"type": "feature_drift", "column": "amount", "metric": "PSI", "value": 0.25, "threshold": 0.2},
                {"type": "feature_drift", "column": "amount", "metric": "PSI", "value": 0.30, "threshold": 0.2},
                {"type": "score_drift_psi", "column": "risk_score", "metric": "PSI", "value": 0.22, "threshold": 0.2},
            ]
        },
        {
            "alerts": [
                {"type": "feature_drift", "column": "step", "metric": "PSI", "value": 0.15, "threshold": 0.2},
            ]
        },
    ]
    summary = summarize_drift_alerts(results)
    assert summary["total_alerts"] == 4
    assert summary["feature_drift_alerts"] == 3
    assert summary["score_drift_alerts"] == 1
    assert summary["most_alerted_features"]["amount"] == 2
    assert summary["alert_types"]["feature_drift"] == 3


def test_summarize_drift_alerts_empty():
    """Empty results should return zero counts."""
    summary = summarize_drift_alerts([])
    assert summary["total_alerts"] == 0
    assert summary["feature_drift_alerts"] == 0


def test_generate_drift_report():
    """Report should contain summary and alert breakdown."""
    results = [
        {
            "alerts": [
                {"type": "feature_drift", "column": "amount", "metric": "PSI", "value": 0.25, "threshold": 0.2},
            ]
        },
    ]
    report = generate_drift_report(results, "test-ds")
    assert "DRIFT MONITORING REPORT: test-ds" in report
    assert "amount" in report
    assert "1 alert(s)" in report


def test_should_retrain_from_drift_psi():
    """PSI above threshold should trigger retraining."""
    result = {"alerts": [{"metric": "PSI", "value": 0.25, "threshold": 0.2}]}
    should, reason = should_retrain_from_drift(result)
    assert should is True
    assert "drift_psi" in reason


def test_should_retrain_from_drift_ks():
    """KS above threshold should trigger retraining."""
    result = {"alerts": [{"metric": "KS", "value": 0.08, "threshold": 0.05}]}
    should, reason = should_retrain_from_drift(result)
    assert should is True
    assert "drift_ks" in reason


def test_should_retrain_from_drift_no_trigger():
    """Below-threshold values should not trigger retraining."""
    result = {"alerts": [{"metric": "PSI", "value": 0.10, "threshold": 0.2}]}
    should, reason = should_retrain_from_drift(result)
    assert should is False
    assert reason == "no_drift_trigger"


def test_should_retrain_from_drift_none():
    """None result should not trigger retraining."""
    should, reason = should_retrain_from_drift(None)
    assert should is False
    assert reason == "no_drift_result"
