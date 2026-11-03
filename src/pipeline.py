"""
End-to-end pipeline runner for the UPI Fraud Detection system.

Ties together all modules (data_prep, models, evaluation, drift_detection,
shap_explainer, retraining) into a single reproducible CLI workflow.

Usage:
    python -m src.pipeline --dataset ieee-cis --run-all
    python -m src.pipeline --dataset paysim --stage train --model lightgbm
    python -m src.pipeline --dataset ieee-cis --stage evaluate --model ieee_lightgbm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Ensure `src` is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import (  # noqa: E402
    DataQualityError,
    handle_class_imbalance,
    load_ieee_cis,
    load_paysim,
    prepare_features_target,
    time_ordered_split,
)
from models import (  # noqa: E402
    MODEL_CONFIGS,
    ModelTrainingError,
    evaluate_model,
    get_model,
    save_model,
    train_model,
)
from drift_detection import (  # noqa: E402
    DriftMonitor,
)
from shap_explainer import (  # noqa: E402
    SHAPExplainer,
)
from retraining import (  # noqa: E402
    DriftTriggeredStrategy,
    RetrainingOrchestrator,
    create_retraining_orchestrator,
)
from evaluation import (  # noqa: E402
    compute_operational_metrics,
    generate_evaluation_report,
    find_optimal_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_SPECS = {
    "ieee-cis": {
        "loader": load_ieee_cis,
        "data_dir": DATA_ROOT / "ieee-cis",
        "time_col": "TransactionDT",
        "target_col": "isFraud",
    },
    "ieee-fraud-detection": {
        "loader": load_ieee_cis,
        "data_dir": DATA_ROOT / "ieee-fraud-detection",
        "time_col": "TransactionDT",
        "target_col": "isFraud",
    },
    "paysim": {
        "loader": load_paysim,
        "data_dir": DATA_ROOT / "paysim",
        "time_col": "step",
        "target_col": "isFraud",
    },
}


class PipelineError(Exception):
    """Raised when the pipeline fails."""


def ensure_dirs() -> None:
    """Create output directories if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset: str) -> Tuple[pd.DataFrame, str, str]:
    """Load a dataset by name. Returns (df, time_col, target_col)."""
    if dataset not in DATASET_SPECS:
        raise PipelineError(
            f"Unknown dataset '{dataset}'. Choose from {list(DATASET_SPECS.keys())}"
        )
    spec = DATASET_SPECS[dataset]
    try:
        df = spec["loader"](spec["data_dir"])
    except FileNotFoundError as exc:
        raise PipelineError(str(exc)) from exc
    except DataQualityError as exc:
        raise PipelineError(f"Data quality check failed: {exc}") from exc
    return df, spec["time_col"], spec["target_col"]


def split_data(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/val/test split."""
    return time_ordered_split(
        df, time_col, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )


def stage_train(
    dataset: str,
    model_name: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    class_weight_method: str = "class_weight",
    save: bool = True,
    quiet: bool = False,
) -> Dict:
    """Train model(s) on a dataset. If model_name is None, trains all baselines."""
    log = print if not quiet else lambda *a, **k: None
    df, time_col, target_col = load_dataset(dataset)
    train_df, val_df, test_df = split_data(df, time_col, train_ratio, val_ratio, test_ratio)

    X_train, y_train = prepare_features_target(train_df, target_col=target_col)
    X_val, y_val = prepare_features_target(val_df, target_col=target_col)
    X_test, y_test = prepare_features_target(test_df, target_col=target_col)

    class_weight_params = handle_class_imbalance(
        X_train, y_train, method=class_weight_method
    )

    results: Dict = {
        "dataset": dataset,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "models": {},
    }

    models_to_train = [model_name] if model_name else list(MODEL_CONFIGS.keys())

    for name in models_to_train:
        log(f"\n=== Training {name} on {dataset} ===")
        try:
            model = get_model(name, class_weight_params)
            model = train_model(model, X_train, y_train, X_val, y_val)
            val_metrics = evaluate_model(model, X_val, y_val)
            test_metrics = evaluate_model(model, X_test, y_test)

            results["models"][name] = {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            log(f"  Val   PR-AUC: {val_metrics['pr_auc']:.4f}  P@100: {val_metrics.get('precision_at_100', 0):.4f}")
            log(f"  Test  PR-AUC: {test_metrics['pr_auc']:.4f}  P@100: {test_metrics.get('precision_at_100', 0):.4f}")

            if save:
                model_path = MODELS_DIR / f"{dataset}_{name}.joblib"
                meta = {
                    "dataset": dataset,
                    "model": name,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "feature_names": X_train.columns.tolist(),
                }
                save_model(model, model_path, metadata=meta)
                log(f"  Saved model -> {model_path}")

        except (ModelTrainingError, Exception) as exc:  # noqa: BLE001
            log(f"  ERROR training {name}: {exc}")
            results["models"][name] = {"error": str(exc)}

    return results


def stage_evaluate(
    dataset: str,
    model_name: str,
    model_path: Optional[Path] = None,
) -> Dict:
    """Evaluate a saved model on the test split."""
    df, time_col, target_col = load_dataset(dataset)
    _, _, test_df = split_data(df, time_col)
    X_test, y_test = prepare_features_target(test_df, target_col=target_col)

    if model_path is None:
        model_path = MODELS_DIR / f"{dataset}_{model_name}.joblib"
    if not model_path.exists():
        raise PipelineError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    metadata = None
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    test_metrics = evaluate_model(model, X_test, y_test)
    best_threshold, threshold_metrics = find_optimal_threshold(
        y_test.values, model.predict_proba(X_test)[:, 1], metric="f1"
    )

    report = generate_evaluation_report(
        model_name=model_name,
        dataset=dataset,
        operational_metrics=test_metrics,
        retraining_metrics={},
        cv_metrics=None,
    )

    result = {
        "dataset": dataset,
        "model": model_name,
        "test_metrics": test_metrics,
        "optimal_threshold": best_threshold,
        "threshold_metrics": threshold_metrics,
        "report": report,
    }

    report_path = REPORTS_DIR / f"{dataset}_{model_name}_evaluation.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Evaluation report -> {report_path}")
    print(report)
    return result


def stage_drift(
    dataset: str,
    reference_window_days: int = 7,
    current_window_days: int = 7,
    score_col: Optional[str] = None,
) -> Dict:
    """Run PSI/KS drift detection between two time windows."""
    df, time_col, target_col = load_dataset(dataset)
    time_days = (df[time_col] // 86400).astype(int) if df[time_col].max() > 1000 else df[time_col]
    min_day, max_day = int(time_days.min()), int(time_days.max())

    ref_end = min_day + reference_window_days
    cur_start = max_day - current_window_days

    ref_mask = (time_days >= min_day) & (time_days < ref_end)
    cur_mask = (time_days >= cur_start) & (time_days <= max_day)

    ref_data = df[ref_mask].copy()
    cur_data = df[cur_mask].copy()

    feature_cols = ref_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != target_col]

    monitor = DriftMonitor()
    monitor.fit(ref_data, feature_cols, score_col=score_col)
    drift_result = monitor.check(cur_data, timestamp=f"day_{ref_end}_to_{max_day}")

    result = {
        "dataset": dataset,
        "reference_window": f"{min_day}-{ref_end}",
        "current_window": f"{cur_start}-{max_day}",
        "ref_size": len(ref_data),
        "cur_size": len(cur_data),
        "drift": drift_result,
        "alerts": drift_result.get("alerts", []),
    }

    alerts_path = REPORTS_DIR / f"{dataset}_drift_alerts.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(alerts_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Drift report -> {alerts_path}")
    print(f"  Alerts: {len(result['alerts'])}")
    for alert in result["alerts"]:
        print(f"    - {alert['type']}: {alert['column']} ({alert['metric']}={alert['value']:.4f})")
    return result


def stage_shap(
    dataset: str,
    model_name: str,
    model_path: Optional[Path] = None,
    max_background: int = 500,
    max_samples: int = 1000,
) -> Dict:
    """Compute SHAP global importance for a model on the test split."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise PipelineError("shap is required for this stage. `pip install shap`") from exc

    df, time_col, target_col = load_dataset(dataset)
    _, _, test_df = split_data(df, time_col)
    train_df, _, _ = split_data(df, time_col, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)

    X_train, _ = prepare_features_target(train_df, target_col=target_col)
    X_test, y_test = prepare_features_target(test_df, target_col=target_col)

    if model_path is None:
        model_path = MODELS_DIR / f"{dataset}_{model_name}.joblib"
    if not model_path.exists():
        raise PipelineError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    explainer = SHAPExplainer(
        model,
        X_train.sample(n=min(max_background, len(X_train)), random_state=42),
        feature_names=X_train.columns.tolist(),
        max_background=max_background,
    )

    importance_df = explainer.global_importance(X_test, max_samples=max_samples)

    scores = model.predict_proba(X_test)[:, 1]
    top_idx = int(np.argsort(scores)[-1])
    instance_exp = explainer.explain_instance(X_test.loc[top_idx].to_frame().T)

    result = {
        "dataset": dataset,
        "model": model_name,
        "global_importance": importance_df.to_dict("records"),
        "instance_index": top_idx,
        "instance_prediction": instance_exp["prediction"],
        "instance_base_value": instance_exp["base_value"],
        "top_contributions": instance_exp["feature_contributions"][:10],
    }

    shap_path = REPORTS_DIR / f"{dataset}_{model_name}_shap_importance.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(shap_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"SHAP report -> {shap_path}")
    print("\nTop 10 global features by mean |SHAP|:")
    print(importance_df.head(10).to_string(index=False))
    return result


def stage_retrain(
    dataset: str,
    model_name: str = "lightgbm",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
    force: bool = False,
    quiet: bool = False,
) -> Dict:
    """Drift-triggered retraining: train baseline, check drift, retrain if triggered.

    Trains an initial model on the train split, runs PSI/KS drift detection
    between train (reference) and test (current), and retrains if drift exceeds
    thresholds or --force is passed. Saves the resulting model.
    """
    log = print if not quiet else lambda *a, **k: None

    df, time_col, target_col = load_dataset(dataset)
    train_df, val_df, test_df = split_data(df, time_col, train_ratio, val_ratio, test_ratio)

    X_train, y_train = prepare_features_target(train_df, target_col=target_col)
    X_val, y_val = prepare_features_target(val_df, target_col=target_col)
    X_test, y_test = prepare_features_target(test_df, target_col=target_col)

    class_weight_params = handle_class_imbalance(X_train, y_train, method="class_weight")

    log(f"\n=== Retraining {model_name} on {dataset} ===")
    model = get_model(model_name, class_weight_params)
    model = train_model(model, X_train, y_train, X_val, y_val)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    # Drift detection: train (reference) vs test (current)
    feature_cols = X_train.columns.tolist()
    monitor = DriftMonitor(psi_threshold=psi_threshold, ks_threshold=ks_threshold)
    monitor.fit(train_df, feature_cols, score_col=None)
    drift_result = monitor.check(test_df, timestamp="retrain_check")

    alerts = drift_result.get("alerts", [])
    drift_detected = len(alerts) > 0

    retrained = False
    trigger: Optional[str] = None
    if force or drift_detected:
        trigger = "force" if force else "drift_detected"
        log(f"  Retraining triggered by: {trigger}")
        model = train_model(model, X_train, y_train, X_val, y_val)
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)
        retrained = True
    else:
        log("  No drift detected — skipping retrain.")

    suffix = "_retrained" if retrained else ""
    model_path = MODELS_DIR / f"{dataset}_{model_name}{suffix}.joblib"
    meta = {
        "dataset": dataset,
        "model": model_name,
        "retrained": retrained,
        "trigger": trigger,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "drift_alerts": alerts,
    }
    save_model(model, model_path, metadata=meta)

    log(f"  Retrained: {retrained}")
    log(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
    log(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    log(f"  Saved model -> {model_path}")

    return {
        "dataset": dataset,
        "model": model_name,
        "retrained": retrained,
        "trigger": trigger,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "drift_alerts": alerts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UPI Fraud Detection end-to-end pipeline",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_SPECS.keys()),
        default="ieee-cis",
        help="Dataset to run against",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output (CI-friendly)",
    )

    sub = parser.add_subparsers(dest="stage", required=True)

    p_train = sub.add_parser("train", help="Train baseline model(s)")
    p_train.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default=None,
                         help="Single model to train (default: all)")
    p_train.add_argument("--no-save", action="store_true", help="Do not persist models")
    p_train.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    p_train.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    p_train.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model on test split")
    p_eval.add_argument("--model", required=True, help="Model name (e.g. lightgbm)")
    p_eval.add_argument("--model-path", type=Path, default=None, help="Path to .joblib model")

    p_drift = sub.add_parser("drift", help="Run PSI/KS drift detection")
    p_drift.add_argument("--ref-days", type=int, default=7, help="Reference window days")
    p_drift.add_argument("--cur-days", type=int, default=7, help="Current window days")
    p_drift.add_argument("--score-col", default=None, help="Optional score column name")

    p_shap = sub.add_parser("shap", help="SHAP global importance + instance explanation")
    p_shap.add_argument("--model", required=True, help="Model name")
    p_shap.add_argument("--model-path", type=Path, default=None)
    p_shap.add_argument("--max-background", type=int, default=500)
    p_shap.add_argument("--max-samples", type=int, default=1000)

    p_retrain = sub.add_parser("retrain", help="Drift-triggered retraining")
    p_retrain.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="lightgbm",
                           help="Model to retrain")
    p_retrain.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    p_retrain.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    p_retrain.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    p_retrain.add_argument("--psi-threshold", type=float, default=0.2, help="PSI drift threshold")
    p_retrain.add_argument("--ks-threshold", type=float, default=0.05, help="KS drift threshold")
    p_retrain.add_argument("--force", action="store_true", help="Force retrain even without drift")

    p_all = sub.add_parser("run-all", help="Run full pipeline: train -> evaluate -> drift -> shap")
    p_all.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="lightgbm",
                       help="Primary model for evaluate/shap stages")
    p_all.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    p_all.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    p_all.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    p_all.add_argument("--psi-threshold", type=float, default=0.2, help="PSI drift threshold")
    p_all.add_argument("--ks-threshold", type=float, default=0.05, help="KS drift threshold")
    p_all.add_argument("--force-retrain", action="store_true", help="Force retrain even without drift")

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    ensure_dirs()

    try:
        if args.stage == "train":
            stage_train(
                dataset=args.dataset,
                model_name=args.model,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                save=not args.no_save,
                quiet=args.quiet,
            )
        elif args.stage == "retrain":
            stage_retrain(
                dataset=args.dataset,
                model_name=args.model,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                psi_threshold=args.psi_threshold,
                ks_threshold=args.ks_threshold,
                force=args.force,
                quiet=args.quiet,
            )
        elif args.stage == "evaluate":
            stage_evaluate(
                dataset=args.dataset,
                model_name=args.model,
                model_path=args.model_path,
            )
        elif args.stage == "drift":
            stage_drift(
                dataset=args.dataset,
                reference_window_days=args.ref_days,
                current_window_days=args.cur_days,
                score_col=args.score_col,
            )
        elif args.stage == "shap":
            stage_shap(
                dataset=args.dataset,
                model_name=args.model,
                model_path=args.model_path,
                max_background=args.max_background,
                max_samples=args.max_samples,
            )
        elif args.stage == "run-all":
            stage_train(
                dataset=args.dataset,
                model_name=args.model,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                save=True,
                quiet=args.quiet,
            )
            stage_evaluate(dataset=args.dataset, model_name=args.model)
            stage_drift(dataset=args.dataset)
            stage_retrain(
                dataset=args.dataset,
                model_name=args.model,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                psi_threshold=args.psi_threshold,
                ks_threshold=args.ks_threshold,
                force=args.force_retrain,
                quiet=args.quiet,
            )
            stage_shap(dataset=args.dataset, model_name=args.model)
        else:
            parser.print_help()
            return 2
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
