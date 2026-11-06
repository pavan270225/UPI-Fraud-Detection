"""
Retraining strategies module for UPI Fraud Detection pipeline.

Implements three retraining strategies:
1. Static - never retrain
2. Scheduled - retrain at fixed intervals
3. Drift-triggered - retrain when drift detected
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RetrainingConfig:
    """Configuration for retraining."""
    strategy: str  # 'static', 'scheduled', 'drift_triggered'
    schedule_days: int = 7  # for scheduled strategy
    psi_threshold: float = 0.2  # for drift-triggered
    ks_threshold: float = 0.05
    min_samples_for_retrain: int = 1000
    max_model_age_days: int = 90  # force retrain after this


@dataclass
class RetrainingEvent:
    """Record of a retraining event."""
    timestamp: str
    trigger: str  # 'scheduled', 'drift_psi', 'drift_ks', 'forced', 'manual'
    model_name: str
    train_window_days: int
    train_samples: int
    val_metrics: Dict[str, float]
    previous_model_age_days: int


@dataclass
class ModelVersion:
    """Record of a saved model version."""
    version_id: str
    timestamp: str
    model_name: str
    trigger: str
    metrics: Dict[str, float]
    file_path: str
    is_active: bool = False


class RetrainingStrategy(ABC):
    """Abstract base class for retraining strategies."""

    @abstractmethod
    def should_retrain(
        self,
        current_day: int,
        last_retrain_day: int,
        drift_result: Optional[Dict] = None,
        model_age_days: int = 0,
    ) -> Tuple[bool, str]:
        """Return (should_retrain, trigger_reason)."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass


class StaticStrategy(RetrainingStrategy):
    """Never retrain after initial training."""

    def should_retrain(self, current_day, last_retrain_day, drift_result=None, model_age_days=0):
        return False, 'static'

    def get_name(self):
        return 'static'


class ScheduledStrategy(RetrainingStrategy):
    """Retrain at fixed intervals."""

    def __init__(self, schedule_days: int = 7):
        self.schedule_days = schedule_days

    def should_retrain(self, current_day, last_retrain_day, drift_result=None, model_age_days=0):
        days_since_retrain = current_day - last_retrain_day
        if days_since_retrain >= self.schedule_days:
            return True, 'scheduled'
        return False, 'scheduled'

    def get_name(self):
        return f'scheduled_{self.schedule_days}d'


class DriftTriggeredStrategy(RetrainingStrategy):
    """Retrain when drift is detected."""

    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.05, max_age_days: int = 90):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.max_age_days = max_age_days

    def should_retrain(self, current_day, last_retrain_day, drift_result=None, model_age_days=0):
        # Force retrain if model too old
        if model_age_days >= self.max_age_days:
            return True, 'forced'

        # Check drift alerts
        if drift_result and drift_result.get('alerts'):
            for alert in drift_result['alerts']:
                if alert['metric'] == 'PSI' and alert['value'] > self.psi_threshold:
                    return True, 'drift_psi'
                if alert['metric'] == 'KS' and alert['value'] > self.ks_threshold:
                    return True, 'drift_ks'

        return False, 'drift_triggered'

    def get_name(self):
        return 'drift_triggered'


class RetrainingOrchestrator:
    """
    Orchestrates retraining across strategies.

    Usage:
        orchestrator = RetrainingOrchestrator(config)
        orchestrator.add_strategy('static', StaticStrategy())
        orchestrator.add_strategy('scheduled', ScheduledStrategy(7))
        orchestrator.add_strategy('drift', DriftTriggeredStrategy())

        # In monitoring loop:
        decision = orchestrator.check_retrain(current_day, last_retrain_day, drift_result)
    """

    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.strategies: Dict[str, RetrainingStrategy] = {}
        self.retraining_history: List[RetrainingEvent] = []
        self.current_model = None
        self.current_model_name = None
        self.last_retrain_day = None
        # Model versioning
        self.models_dir: Path = PROJECT_ROOT / "models" / "versions"
        self.versions: List[ModelVersion] = []
        self.active_version_id: Optional[str] = None
        self._load_version_registry()

    def add_strategy(self, name: str, strategy: RetrainingStrategy):
        """Register a retraining strategy."""
        self.strategies[name] = strategy

    def check_retrain(
        self,
        current_day: int,
        drift_result: Optional[Dict] = None,
    ) -> Tuple[bool, str, str]:
        """
        Check if any strategy wants to retrain.

        Returns:
            (should_retrain, strategy_name, trigger_reason)
        """
        if self.last_retrain_day is None:
            return True, 'initial', 'initial_training'

        model_age = current_day - self.last_retrain_day

        # Check active strategy
        active_strategy = self.strategies.get(self.config.strategy)
        if active_strategy:
            should, trigger = active_strategy.should_retrain(
                current_day, self.last_retrain_day, drift_result, model_age
            )
            if should:
                return True, active_strategy.get_name(), trigger

        return False, self.config.strategy, 'no_trigger'

    def execute_retrain(
        self,
        train_fn: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        current_day: int,
        trigger: str,
        model_name: str,
        train_window_days: int,
    ) -> Any:
        """
        Execute retraining using provided training function.

        Args:
            train_fn: Function(model, X_train, y_train, X_val, y_val) -> trained_model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            current_day: Current simulation day
            trigger: Trigger reason
            model_name: Model identifier
            train_window_days: Size of training window

        Returns:
            New trained model
        """
        print(f"  Retraining triggered by: {trigger} (day {current_day})")

        # Train new model
        if self.current_model is not None:
            new_model = train_fn(self.current_model, X_train, y_train, X_val, y_val)
        else:
            # Initial training - train_fn should handle None model
            new_model = train_fn(None, X_train, y_train, X_val, y_val)

        # Evaluate
        from models import evaluate_model
        val_metrics = evaluate_model(new_model, X_val, y_val)

        # Record event
        event = RetrainingEvent(
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            model_name=model_name,
            train_window_days=train_window_days,
            train_samples=len(X_train),
            val_metrics=val_metrics,
            previous_model_age_days=current_day - self.last_retrain_day if self.last_retrain_day else 0,
        )
        self.retraining_history.append(event)

        # Save versioned model to disk
        from models import save_model as _save_model

        self.models_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_{ts}"
        model_path = self.models_dir / f"{version_id}.joblib"
        version_metadata = {
            "version_id": version_id,
            "timestamp": event.timestamp,
            "model_name": model_name,
            "trigger": trigger,
            "metrics": val_metrics,
            "train_samples": len(X_train),
            "previous_model_age_days": event.previous_model_age_days,
        }
        _save_model(new_model, model_path, metadata=version_metadata)

        # Register version
        for v in self.versions:
            v.is_active = False
        new_version = ModelVersion(
            version_id=version_id,
            timestamp=event.timestamp,
            model_name=model_name,
            trigger=trigger,
            metrics=val_metrics,
            file_path=str(model_path),
            is_active=True,
        )
        self.versions.append(new_version)
        self.active_version_id = version_id
        self._save_version_registry()

        # Update state
        self.current_model = new_model
        self.current_model_name = model_name
        self.last_retrain_day = current_day

        print(f"  Retrained {model_name}: PR-AUC={val_metrics['pr_auc']:.4f}, P@100={val_metrics.get('precision_at_100', 0):.4f}")
        print(f"  Saved version {version_id} -> {model_path}")

        return new_model

    def get_history_df(self) -> pd.DataFrame:
        """Get retraining history as DataFrame."""
        if not self.retraining_history:
            return pd.DataFrame()

        records = []
        for event in self.retraining_history:
            record = {
                'timestamp': event.timestamp,
                'trigger': event.trigger,
                'model': event.model_name,
                'train_window_days': event.train_window_days,
                'train_samples': event.train_samples,
                'model_age_days': event.previous_model_age_days,
            }
            record.update(event.val_metrics)
            records.append(record)

        return pd.DataFrame(records)

    # --- Model versioning / rollback ---

    def _registry_path(self) -> Path:
        """Return the path to the version registry JSON file."""
        return self.models_dir / "versions.json"

    def _save_version_registry(self) -> None:
        """Persist the version registry to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for v in self.versions:
            records.append(
                {
                    "version_id": v.version_id,
                    "timestamp": v.timestamp,
                    "model_name": v.model_name,
                    "trigger": v.trigger,
                    "metrics": v.metrics,
                    "file_path": v.file_path,
                    "is_active": v.is_active,
                }
            )
        with open(self._registry_path(), "w") as f:
            json.dump(
                {"active_version_id": self.active_version_id, "versions": records},
                f,
                indent=2,
                default=str,
            )

    def _load_version_registry(self) -> None:
        """Load the version registry from disk if it exists."""
        reg_path = self._registry_path()
        if not reg_path.exists():
            return
        try:
            with open(reg_path) as f:
                data = json.load(f)
            self.active_version_id = data.get("active_version_id")
            for v in data.get("versions", []):
                self.versions.append(
                    ModelVersion(
                        version_id=v["version_id"],
                        timestamp=v["timestamp"],
                        model_name=v["model_name"],
                        trigger=v["trigger"],
                        metrics=v.get("metrics", {}),
                        file_path=v["file_path"],
                        is_active=v.get("is_active", False),
                    )
                )
        except Exception:  # noqa: BLE001
            # If the registry is corrupted, start fresh rather than crash.
            self.versions = []
            self.active_version_id = None

    def list_versions(self) -> List[ModelVersion]:
        """Return all saved model versions, newest first."""
        return sorted(self.versions, key=lambda v: v.timestamp, reverse=True)

    def get_active_version(self) -> Optional[ModelVersion]:
        """Return the currently active model version, or None."""
        if self.active_version_id is None:
            return None
        for v in self.versions:
            if v.version_id == self.active_version_id:
                return v
        return None

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Return a specific model version by ID, or None if not found."""
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None

    def rollback(self, version_id: str) -> Any:
        """
        Roll back to a previous model version.

        Sets the specified version as active, loads its model from disk,
        and updates orchestrator state. Returns the loaded model.

        Args:
            version_id: The version ID to roll back to.

        Returns:
            The rolled-back model object.

        Raises:
            ValueError: If the version ID is not found.
            FileNotFoundError: If the model file for the version is missing.
        """
        from models import load_model as _load_model

        target = None
        for v in self.versions:
            if v.version_id == version_id:
                target = v
                break
        if target is None:
            raise ValueError(f"Model version not found: {version_id}")

        model_path = Path(target.file_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Deactivate all versions, activate the target, persist registry
        for v in self.versions:
            v.is_active = False
        target.is_active = True
        self.active_version_id = version_id
        self._save_version_registry()

        # Load the model
        model, _ = _load_model(model_path)
        self.current_model = model
        self.current_model_name = target.model_name

        print(f"  Rolled back to version {version_id} ({target.model_name})")
        return model

    def compare_strategies(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        time_col: str,
        train_fn: Callable,
        train_window_days: int = 30,
        test_window_days: int = 7,
    ) -> pd.DataFrame:
        """
        Run simulation comparing all three strategies on historical data.

        Returns:
            DataFrame with results per strategy per window
        """
        # Convert time to days
        time_days = (data[time_col] // 86400).astype(int) if data[time_col].max() > 10000 else data[time_col]
        min_day, max_day = time_days.min(), time_days.max()

        strategies = {
            'static': StaticStrategy(),
            'scheduled': ScheduledStrategy(self.config.schedule_days),
            'drift_triggered': DriftTriggeredStrategy(self.config.psi_threshold, self.config.ks_threshold),
        }

        results = []

        for strategy_name, strategy in strategies.items():
            print(f"\nSimulating {strategy_name} strategy...")

            # Reset orchestrator for this strategy
            self.strategies = {'active': strategy}
            self.config.strategy = 'active'
            self.current_model = None
            self.last_retrain_day = None
            self.retraining_history = []

            # Initial training on first window
            initial_mask = (time_days >= min_day) & (time_days < min_day + train_window_days)
            initial_data = data[initial_mask].copy()

            if len(initial_data) < self.config.min_samples_for_retrain:
                continue

            from data_prep import prepare_features_target
            X_init, y_init = prepare_features_target(initial_data)

            # Simple split for initial validation
            split_idx = int(len(X_init) * 0.8)
            X_train_init, y_train_init = X_init.iloc[:split_idx], y_init.iloc[:split_idx]
            X_val_init, y_val_init = X_init.iloc[split_idx:], y_init.iloc[split_idx:]

            self.execute_retrain(
                train_fn, X_train_init, y_train_init, X_val_init, y_val_init,
                min_day, 'initial', 'model', train_window_days
            )

            # Rolling evaluation
            for window_start in range(min_day + train_window_days, max_day - test_window_days, test_window_days):
                window_end = window_start + test_window_days
                train_start = window_start - train_window_days

                # Training data for this window
                train_mask = (time_days >= train_start) & (time_days < window_start)
                train_data = data[train_mask].copy()

                # Test data
                test_mask = (time_days >= window_start) & (time_days < window_end)
                test_data = data[test_mask].copy()

                if len(test_data) == 0:
                    continue

                # Get drift result from previous window (simplified)
                # In practice, use DriftMonitor on train_data vs previous train_data
                drift_result = None  # Would be computed in real implementation

                # Check retrain
                should_retrain, _, trigger = self.check_retrain(window_start, drift_result)

                if should_retrain and len(train_data) >= self.config.min_samples_for_retrain:
                    X_train, y_train = prepare_features_target(train_data)
                    split_idx = int(len(X_train) * 0.8)
                    X_tr, y_tr = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
                    X_val, y_val = X_train.iloc[split_idx:], y_train.iloc[split_idx:]

                    self.execute_retrain(
                        train_fn, X_tr, y_tr, X_val, y_val,
                        window_start, trigger, 'model', train_window_days
                    )

                # Evaluate current model on test window
                if self.current_model is not None:
                    X_test, y_test = prepare_features_target(test_data)
                    from models import evaluate_model
                    metrics = evaluate_model(self.current_model, X_test, y_test)

                    results.append({
                        'strategy': strategy_name,
                        'window_start': window_start,
                        'window_end': window_end,
                        'retrained': should_retrain,
                        'trigger': trigger if should_retrain else 'none',
                        **metrics,
                    })

        return pd.DataFrame(results)


def create_retraining_orchestrator(
    strategy: str = 'drift_triggered',
    schedule_days: int = 7,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
) -> RetrainingOrchestrator:
    """Factory function to create configured orchestrator."""
    config = RetrainingConfig(
        strategy=strategy,
        schedule_days=schedule_days,
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold,
    )
    orchestrator = RetrainingOrchestrator(config)

    orchestrator.add_strategy('static', StaticStrategy())
    orchestrator.add_strategy('scheduled', ScheduledStrategy(schedule_days))
    orchestrator.add_strategy('drift_triggered', DriftTriggeredStrategy(psi_threshold, ks_threshold))

    return orchestrator


if __name__ == '__main__':
    print("Retraining strategies module ready")
    print("Strategies: Static, Scheduled, DriftTriggered")