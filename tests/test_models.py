"""Unit tests for models module — model factory, training, evaluation, hyperparameter tuning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from models import (
    MODEL_CONFIGS,
    MODEL_PARAM_GRIDS,
    ModelTrainingError,
    evaluate_model,
    get_model,
    load_model,
    save_model,
    train_model,
    tune_hyperparameters,
)


def test_get_model_all_types():
    """get_model should return a valid model for every configured type."""
    for name in MODEL_CONFIGS:
        model = get_model(name)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")


def test_get_model_unknown_type_raises():
    """get_model should raise ValueError for unknown model name."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model")


def test_model_param_grids_covers_all_configs():
    """MODEL_PARAM_GRIDS must define a grid for every model in MODEL_CONFIGS."""
    for name in MODEL_CONFIGS:
        assert name in MODEL_PARAM_GRIDS, f"Missing param grid for {name}"
        assert len(MODEL_PARAM_GRIDS[name]) > 0


def test_train_model_logistic_regression():
    """train_model should fit a logistic regression model successfully."""
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "f2": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    model = get_model("logistic_regression")
    trained = train_model(model, X, y)
    assert trained is not None
    assert hasattr(trained, "predict_proba")


def test_train_model_training_error():
    """train_model should raise ModelTrainingError on invalid input."""
    model = get_model("logistic_regression")
    X = pd.DataFrame({"f1": [0.1]})
    y = pd.Series([0])
    with pytest.raises(ModelTrainingError):
        train_model(model, X.iloc[:0], y.iloc[:0])


def test_evaluate_model_returns_metrics():
    """evaluate_model should return dict with expected metric keys."""
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    model = get_model("logistic_regression")
    model.fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_tune_hyperparameters_random_search():
    """tune_hyperparameters with random search should return best model and params."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    best_model, best_params = tune_hyperparameters(
        "logistic_regression", X, y, search_type="random", n_iter=5, cv=3
    )
    assert best_model is not None
    assert best_params is not None
    assert isinstance(best_params, dict)
    assert hasattr(best_model, "predict_proba")


def test_tune_hyperparameters_grid_search():
    """tune_hyperparameters with grid search should return best model and params."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    best_model, best_params = tune_hyperparameters(
        "random_forest", X, y, search_type="grid", cv=2
    )
    assert best_model is not None
    assert best_params is not None
    assert isinstance(best_params, dict)


def test_tune_hyperparameters_unknown_model():
    """tune_hyperparameters should raise ValueError for unknown model name."""
    X = pd.DataFrame({"f1": [0.1, 0.2]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError, match="param grid"):
        tune_hyperparameters("nonexistent_model", X, y)


def test_tune_hyperparameters_invalid_search_type():
    """tune_hyperparameters should raise ValueError for invalid search_type."""
    X = pd.DataFrame({"f1": [0.1, 0.2]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError, match="Unknown search_type"):
        tune_hyperparameters("logistic_regression", X, y, search_type="invalid")


def test_save_and_load_model(tmp_path):
    """save_model and load_model should round-trip a model with metadata."""
    model_path = tmp_path / "test_model.joblib"
    model = LogisticRegression()
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    model.fit(X, y)
    metadata = {"version_id": "v1", "pr_auc": 0.85}
    save_model(model, model_path, metadata=metadata)
    assert model_path.exists()
    assert model_path.with_suffix(".json").exists()
    loaded, meta = load_model(model_path)
    assert loaded is not None
    assert meta is not None
    assert meta["version_id"] == "v1"
    assert meta["pr_auc"] == 0.85
