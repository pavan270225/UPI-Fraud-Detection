"""Unit tests for retraining module — strategies, orchestrator, versioning/rollback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
import pytest
import time
from sklearn.linear_model import LogisticRegression

from retraining import (
    DriftTriggeredStrategy,
    ModelVersion,
    RetrainingConfig,
    RetrainingEvent,
    RetrainingOrchestrator,
    ScheduledStrategy,
    StaticStrategy,
)


def test_model_version_dataclass():
    """ModelVersion should store all fields and default is_active to False."""
    v = ModelVersion(
        version_id="v1",
        timestamp="2026-01-01T00:00:00",
        model_name="lightgbm",
        trigger="initial",
        metrics={"pr_auc": 0.85},
        file_path="/models/v1.joblib",
    )
    assert v.version_id == "v1"
    assert v.model_name == "lightgbm"
    assert v.trigger == "initial"
    assert v.metrics == {"pr_auc": 0.85}
    assert v.file_path == "/models/v1.joblib"
    assert v.is_active is False


def test_model_version_active_default():
    """ModelVersion is_active should default to False."""
    v = ModelVersion(
        version_id="v1",
        timestamp="2026-01-01",
        model_name="lgb",
        trigger="initial",
        metrics={},
        file_path="/v1.joblib",
    )
    assert v.is_active is False


def test_retraining_event_dataclass():
    """RetrainingEvent should store all fields."""
    e = RetrainingEvent(
        timestamp="2026-01-01T00:00:00",
        trigger="drift_psi",
        model_name="lightgbm",
        train_window_days=30,
        train_samples=1000,
        val_metrics={"pr_auc": 0.85},
        previous_model_age_days=7,
    )
    assert e.trigger == "drift_psi"
    assert e.model_name == "lightgbm"
    assert e.train_samples == 1000


def test_static_strategy_never_retrains():
    """StaticStrategy should never trigger retraining."""
    strategy = StaticStrategy()
    should, trigger = strategy.should_retrain(100, 50)
    assert should is False
    assert trigger == "static"
    assert strategy.get_name() == "static"


def test_scheduled_strategy():
    """ScheduledStrategy should trigger after schedule_days."""
    strategy = ScheduledStrategy(schedule_days=7)
    assert strategy.should_retrain(6, 0) == (False, "scheduled")   # 6 < 7 days
    assert strategy.should_retrain(7, 0) == (True, "scheduled")    # 7 >= 7 days
    assert strategy.get_name() == "scheduled_7d"


def test_drift_strategy_triggers_on_psi():
    """DriftTriggeredStrategy should trigger on PSI above threshold."""
    strategy = DriftTriggeredStrategy(psi_threshold=0.2, ks_threshold=0.05)
    drift_result = {"alerts": [{"metric": "PSI", "value": 0.3}]}
    should, trigger = strategy.should_retrain(10, 5, drift_result)
    assert should is True
    assert trigger == "drift_psi"


def test_drift_strategy_triggers_on_ks():
    """DriftTriggeredStrategy should trigger on KS above threshold."""
    strategy = DriftTriggeredStrategy(psi_threshold=0.2, ks_threshold=0.05)
    drift_result = {"alerts": [{"metric": "KS", "value": 0.1}]}
    should, trigger = strategy.should_retrain(10, 5, drift_result)
    assert should is True
    assert trigger == "drift_ks"


def test_drift_strategy_no_trigger_below_threshold():
    """DriftTriggeredStrategy should not trigger below thresholds."""
    strategy = DriftTriggeredStrategy(psi_threshold=0.2, ks_threshold=0.05)
    drift_result = {"alerts": [{"metric": "PSI", "value": 0.1}, {"metric": "KS", "value": 0.01}]}
    should, trigger = strategy.should_retrain(10, 5, drift_result)
    assert should is False
    assert trigger == "drift_triggered"


def test_drift_strategy_forced_by_max_age():
    """DriftTriggeredStrategy should force retrain when model too old."""
    strategy = DriftTriggeredStrategy(psi_threshold=0.2, ks_threshold=0.05, max_age_days=90)
    should, trigger = strategy.should_retrain(100, 0, None, model_age_days=90)
    assert should is True
    assert trigger == "forced"


def test_orchestrator_initial_retrain():
    """Orchestrator with no last_retrain_day should trigger initial training."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    should, strategy, trigger = orch.check_retrain(0)
    assert should is True
    assert trigger == "initial_training"


def test_orchestrator_no_trigger():
    """Orchestrator should return no_trigger when strategy doesn't want retrain."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    orch.last_retrain_day = 50
    should, strategy, trigger = orch.check_retrain(60)
    assert should is False
    assert trigger == "no_trigger"


def test_orchestrator_list_versions_newest_first():
    """list_versions should return versions sorted newest first."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    v1 = ModelVersion("v1", "2026-01-01", "lgb", "initial", {"pr_auc": 0.8}, "/v1.joblib", True)
    v2 = ModelVersion("v2", "2026-01-02", "lgb", "drift", {"pr_auc": 0.85}, "/v2.joblib", False)
    orch.versions = [v1, v2]
    versions = orch.list_versions()
    assert len(versions) == 2
    assert versions[0].version_id == "v2"  # newest first


def test_orchestrator_get_active_version():
    """get_active_version should return the active version."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    v1 = ModelVersion("v1", "2026-01-01", "lgb", "initial", {"pr_auc": 0.8}, "/v1.joblib", True)
    v2 = ModelVersion("v2", "2026-01-02", "lgb", "drift", {"pr_auc": 0.85}, "/v2.joblib", False)
    orch.versions = [v1, v2]
    orch.active_version_id = "v1"
    active = orch.get_active_version()
    assert active.version_id == "v1"

    # No active version set
    orch2 = RetrainingOrchestrator(config)
    assert orch2.get_active_version() is None


def test_orchestrator_get_version():
    """get_version should return a specific version by ID."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    v1 = ModelVersion("v1", "2026-01-01", "lgb", "initial", {"pr_auc": 0.8}, "/v1.joblib", True)
    orch.versions = [v1]
    v = orch.get_version("v1")
    assert v is not None
    assert v.version_id == "v1"
    assert orch.get_version("nonexistent") is None


def test_orchestrator_rollback_success(tmp_path):
    """rollback should switch active version and load model from disk."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    orch.models_dir = tmp_path / "versions"
    orch.models_dir.mkdir(parents=True, exist_ok=True)

    # Train and save two dummy models
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression().fit(X, y)
    model_dir = orch.models_dir
    joblib.dump(m1, model_dir / "v1.joblib")
    joblib.dump(m2, model_dir / "v2.joblib")

    v1 = ModelVersion("v1", "2026-01-01", "lgb", "initial", {"pr_auc": 0.8}, str(model_dir / "v1.joblib"), True)
    v2 = ModelVersion("v2", "2026-01-02", "lgb", "drift", {"pr_auc": 0.85}, str(model_dir / "v2.joblib"), False)
    orch.versions = [v1, v2]
    orch.active_version_id = "v1"

    model = orch.rollback("v2")
    active = orch.get_active_version()
    assert active.version_id == "v2"
    assert v2.is_active is True
    assert v1.is_active is False
    assert model is not None


def test_orchestrator_rollback_nonexistent():
    """rollback should raise ValueError for unknown version ID."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    with pytest.raises(ValueError, match="not found"):
        orch.rollback("nonexistent")


def test_orchestrator_rollback_missing_file_preserves_state(tmp_path):
    """rollback should raise FileNotFoundError if model file missing and preserve state."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    orch.models_dir = tmp_path / "versions"
    orch.models_dir.mkdir(parents=True, exist_ok=True)

    v1 = ModelVersion("v1", "2026-01-01", "lgb", "initial", {"pr_auc": 0.8}, "/v1.joblib", True)
    v2 = ModelVersion("v2", "2026-01-02", "lgb", "drift", {"pr_auc": 0.85}, "/missing.joblib", False)
    orch.versions = [v1, v2]
    orch.active_version_id = "v1"

    with pytest.raises(FileNotFoundError):
        orch.rollback("v2")

    # State should be preserved (still v1 active)
    active = orch.get_active_version()
    assert active.version_id == "v1"


def test_execute_retrain_saves_version(tmp_path):
    """execute_retrain should save a versioned model and register it."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    orch.models_dir = tmp_path / "versions"

    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    X_val = pd.DataFrame({"f1": [0.15, 0.25], "f2": [2, 3]})
    y_val = pd.Series([0, 1])

    def train_fn(m, X_train, y_train, X_val, y_val):
        return LogisticRegression().fit(X_train, y_train)

    new_model = orch.execute_retrain(
        train_fn, X, y, X_val, y_val, current_day=1, trigger="initial",
        model_name="logistic_regression", train_window_days=30
    )
    assert new_model is not None
    assert len(orch.versions) == 1
    assert orch.active_version_id is not None
    assert orch.get_active_version() is not None
    assert orch.active_version_id.startswith("logistic_regression_")


def test_execute_retrain_deactivates_previous(tmp_path):
    """execute_retrain should deactivate previous active version."""
    config = RetrainingConfig(strategy="static")
    orch = RetrainingOrchestrator(config)
    orch.models_dir = tmp_path / "versions"

    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    X_val = pd.DataFrame({"f1": [0.15, 0.25], "f2": [2, 3]})
    y_val = pd.Series([0, 1])

    def train_fn(m, X_train, y_train, X_val, y_val):
        return LogisticRegression().fit(X_train, y_train)

    # First retrain
    orch.execute_retrain(
        train_fn, X, y, X_val, y_val, current_day=1, trigger="initial",
        model_name="logistic_regression", train_window_days=30
    )
    first_active = orch.active_version_id
    assert orch.versions[0].is_active is True

    # Wait to ensure different timestamp for second version
    time.sleep(1.1)

    # Second retrain
    orch.execute_retrain(
        train_fn, X, y, X_val, y_val, current_day=2, trigger="drift_psi",
        model_name="logistic_regression", train_window_days=30
    )
    # First version should be deactivated, second activated
    assert orch.versions[0].is_active is False
    assert orch.versions[1].is_active is True
    assert orch.active_version_id != first_active
