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
