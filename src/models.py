"""
Model training module for UPI Fraud Detection pipeline.

Provides model factory, training loop, and persistence for baseline models.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import inspect
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb




class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


MODEL_CONFIGS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'n_jobs': -1,
            'random_state': 42,
        },
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'n_jobs': -1,
            'random_state': 42,
        },
    },
    'xgboost': {
        'class': xgb.XGBClassifier,
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'aucpr',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0,
        },
    },
    'lightgbm': {
        'class': lgb.LGBMClassifier,
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'metric': 'aucpr',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1,
        },
    },
}


# Hyperparameter search grids for each model type.
# Keys must match MODEL_CONFIGS model names.
MODEL_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    'logistic_regression': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['lbfgs', 'liblinear'],
        'class_weight': [None, 'balanced'],
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
    'xgboost': {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    },
    'lightgbm': {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    },
}


def get_model(model_name: str, class_weight_params: Optional[Dict] = None) -> Any:
    """Factory function to get model instance with configured params."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_name]
    params = config['params'].copy()
    if class_weight_params:
        # Only pass class_weight_params keys that the model class actually accepts.
        # e.g. LogisticRegression accepts class_weight but not scale_pos_weight;
        # XGBClassifier/LGBMClassifier accept scale_pos_weight via **kwargs.
        init_sig = inspect.signature(config['class'].__init__)
        valid_keys = set(init_sig.parameters.keys())
        has_kwargs = any(
            p.kind == p.VAR_KEYWORD for p in init_sig.parameters.values()
        )
        if has_kwargs:
            params.update(class_weight_params)
        else:
            filtered_params = {
                k: v for k, v in class_weight_params.items() if k in valid_keys
            }
            params.update(filtered_params)
    return config['class'](**params)


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50,
) -> Any:
    """Train model with optional validation for early stopping."""
    model_name = type(model).__name__
    try:
        if model_name in ['XGBClassifier', 'LGBMClassifier'] and X_val is not None:
            eval_set = [(X_val, y_val)]
            if model_name == 'XGBClassifier':
                # XGBoost 3.x: early_stopping_rounds/callbacks removed from sklearn fit()
                model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                model.fit(X_train, y_train, eval_set=eval_set,
                          eval_metric='auc',
                          callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)])
        else:
            model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ModelTrainingError(f"Training failed for {model_name}: {e}")


def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = 'random',
    n_iter: int = 30,
    cv: int = 3,
    scoring=None,
    class_weight_params: Optional[Dict] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """Tune hyperparameters for a model using cross-validated search.

    Args:
        model_name: Model key from MODEL_CONFIGS/MODEL_PARAM_GRIDS.
        X_train: Training features.
        y_train: Training labels.
        search_type: 'grid' for exhaustive grid search, 'random' for randomized search.
        n_iter: Number of parameter settings sampled (random search only).
        cv: Number of cross-validation folds.
        scoring: Scorer callable; defaults to PR-AUC.
        class_weight_params: Class imbalance params passed to get_model.
        random_state: Random seed for reproducibility.
        n_jobs: Parallel jobs for cross-validation.

    Returns:
        (best_model, best_params) where best_model is the fitted estimator
        with the best found hyperparameters and best_params is the param dict.
    """
    if model_name not in MODEL_PARAM_GRIDS:
        raise ValueError(
            f"No param grid defined for model: {model_name}. "
            f"Available: {list(MODEL_PARAM_GRIDS.keys())}"
        )

    if scoring is None:
        scoring = 'average_precision'

    base_model = get_model(model_name, class_weight_params)
    param_grid = MODEL_PARAM_GRIDS[model_name]

    if search_type == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
        )
    elif search_type == 'random':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type}. Use 'grid' or 'random'.")

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    print(f"  Best {model_name} params: {best_params}")
    print(f"  Best CV PR-AUC: {search.best_score_:.4f}")

    return best_model, best_params


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    k_values: List[int] = [50, 100, 200, 500],
) -> Dict[str, float]:
    """Evaluate model with multiple metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        'pr_auc': average_precision_score(y, y_pred_proba),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
    }
    n_samples = len(y)
    for k in k_values:
        if k <= n_samples:
            top_k_idx = np.argsort(y_pred_proba)[-k:]
            metrics[f'precision_at_{k}'] = precision_score(y.iloc[top_k_idx], np.ones(k), zero_division=0)
            metrics[f'recall_at_{k}'] = recall_score(y.iloc[top_k_idx], np.ones(k), zero_division=0)
    return metrics


def rolling_time_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    train_window: int = 30,
    test_window: int = 7,
    time_col: str = 'TransactionDT',
    class_weight_params: Optional[Dict] = None,
) -> List[Dict[str, float]]:
    """Rolling time-window cross-validation."""
    if time_col not in X.columns:
        raise ValueError(f"Time column '{time_col}' not in features")
    time_days = (X[time_col] // 86400).astype(int)
    min_day, max_day = time_days.min(), time_days.max()
    total_days = max_day - min_day
    step = (total_days - train_window - test_window) // (n_splits - 1) if n_splits > 1 else 0
    results = []
    for fold in range(n_splits):
        train_start = min_day + fold * step
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window
        train_mask = (time_days >= train_start) & (time_days < train_end)
        test_mask = (time_days >= test_start) & (time_days < test_end)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_train, y_train = X[train_mask].drop(columns=[time_col]), y[train_mask]
        X_test, y_test = X[test_mask].drop(columns=[time_col]), y[test_mask]
        model = get_model(model_name, class_weight_params)
        model = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics['fold'] = fold
        metrics['train_days'] = f"{train_start}-{train_end}"
        metrics['test_days'] = f"{test_start}-{test_end}"
        results.append(metrics)
        print(f"  Fold {fold}: PR-AUC={metrics['pr_auc']:.4f}, P@100={metrics.get('precision_at_100', 0):.4f}")
    return results


def save_model(model, path: Path, metadata: Optional[Dict] = None) -> None:
    """Save model and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    if metadata:
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def load_model(path: Path) -> Tuple[Any, Optional[Dict]]:
    """Load model and metadata."""
    model = joblib.load(path)
    meta_path = path.with_suffix('.json')
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    return model, metadata


def train_all_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_weight_params: Optional[Dict] = None,
    time_col: str = 'TransactionDT',
) -> Dict[str, Dict]:
    """Train all baseline models and return results."""
    results = {}
    for model_name in MODEL_CONFIGS.keys():
        print(f"\nTraining {model_name}...")
        model = get_model(model_name, class_weight_params)
        model = train_model(model, X_train, y_train, X_val, y_val)
        val_metrics = evaluate_model(model, X_val, y_val)
        results[model_name] = {'model': model, 'val_metrics': val_metrics}
        print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"  Val P@100: {val_metrics.get('precision_at_100', 'N/A'):.4f}")
        print(f"  Val R@100: {val_metrics.get('recall_at_100', 'N/A'):.4f}")
    return results


if __name__ == '__main__':
    print("Model configs available:", list(MODEL_CONFIGS.keys()))
    for name in MODEL_CONFIGS:
        m = get_model(name)
        print(f"  {name}: {type(m).__name__}")