"""
SHAP explainer module for UPI Fraud Detection pipeline.

Provides per-instance explanations (TreeSHAP) and global feature importance tracking
over time windows to correlate with drift alerts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


class SHAPExplainerError(Exception):
    """Raised when SHAP explanation fails."""
    pass


class SHAPExplainer:
    """Wrapper for SHAP TreeExplainer with drift tracking."""

    def __init__(
        self,
        model: Any,
        X_background: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        max_background: int = 1000,
    ):
        if not SHAP_AVAILABLE:
            raise SHAPExplainerError("SHAP not installed. Run: pip install shap")

        self.model = model
        self.feature_names = feature_names or X_background.columns.tolist()

        if len(X_background) > max_background:
            X_background = X_background.sample(n=max_background, random_state=42)

        self.X_background = X_background
        self.explainer = shap.TreeExplainer(model, X_background)
        self.importance_history = []

    def explain_instance(
        self,
        X_instance: pd.DataFrame,
        max_display: int = 20,
    ) -> Dict[str, Any]:
        """Explain a single instance (or batch)."""
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_frame().T

        X_instance = X_instance[self.feature_names]
        shap_values = self.explainer.shap_values(X_instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if isinstance(self.explainer.expected_value, list):
            base_value = self.explainer.expected_value[1]
        else:
            base_value = self.explainer.expected_value

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        n_samples = shap_values.shape[0]
        results = []

        for i in range(n_samples):
            contributions = list(zip(self.feature_names, shap_values[i]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            results.append({
                'base_value': float(base_value),
                'shap_values': shap_values[i].tolist(),
                'feature_contributions': contributions[:max_display],
                'prediction': float(base_value + shap_values[i].sum()),
            })

        return results[0] if n_samples == 1 else results

    def explain_batch(
        self,
        X_batch: pd.DataFrame,
        max_samples: int = 1000,
    ) -> np.ndarray:
        """Get SHAP values for a batch."""
        if len(X_batch) > max_samples:
            X_batch = X_batch.sample(n=max_samples, random_state=42)

        X_batch = X_batch[self.feature_names]
        shap_values = self.explainer.shap_values(X_batch)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values

    def global_importance(
        self,
        X_data: pd.DataFrame,
        max_samples: int = 1000,
    ) -> pd.DataFrame:
        """Calculate global feature importance (mean |SHAP|)."""
        shap_values = self.explain_batch(X_data, max_samples)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        std_shap = np.std(shap_values, axis=0)
        mean_shap = np.mean(shap_values, axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs,
            'std_shap': std_shap,
            'mean_shap': mean_shap,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        return importance_df

    def track_importance_drift(
        self,
        importance_df: pd.DataFrame,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track feature importance over time and detect drift."""
        if timestamp is None:
            timestamp = pd.Timestamp.now().isoformat()

        importance_dict = importance_df.set_index('feature')['mean_abs_shap'].to_dict()
        self.importance_history.append({
            'timestamp': timestamp,
            'importance': importance_dict,
        })

        result = {
            'timestamp': timestamp,
            'current_importance': importance_dict,
            'drift_from_previous': None,
        }

        if len(self.importance_history) >= 2:
            prev_importance = self.importance_history[-2]['importance']
            common_features = set(importance_dict.keys()) & set(prev_importance.keys())

            if common_features:
                curr_rank = pd.Series({f: importance_dict[f] for f in common_features}).rank()
                prev_rank = pd.Series({f: prev_importance[f] for f in common_features}).rank()
                spearman = curr_rank.corr(prev_rank, method='spearman')

                top_k = 20
                curr_top = set(pd.Series(importance_dict).nlargest(top_k).index)
                prev_top = set(pd.Series(prev_importance).nlargest(top_k).index)
                top_k_overlap = len(curr_top & prev_top) / top_k

                result['drift_from_previous'] = {
                    'spearman_correlation': float(spearman) if not np.isnan(spearman) else 0.0,
                    'top_k_overlap': top_k_overlap,
                    'features_dropped_from_topk': list(prev_top - curr_top),
                    'features_entered_topk': list(curr_top - prev_top),
                }

        return result

    def get_importance_history(self) -> pd.DataFrame:
        """Get importance history as DataFrame."""
        if not self.importance_history:
            return pd.DataFrame()

        records = []
        for entry in self.importance_history:
            for feat, imp in entry['importance'].items():
                records.append({
                    'timestamp': entry['timestamp'],
                    'feature': feat,
                    'importance': imp,
                })
        return pd.DataFrame(records)

    def plot_waterfall(
        self,
        X_instance: pd.DataFrame,
        max_display: int = 20,
        show: bool = True,
    ) -> Any:
        """Generate SHAP waterfall plot for an instance."""
        if not SHAP_AVAILABLE:
            raise SHAPExplainerError("SHAP not available")

        X_instance = X_instance[self.feature_names]
        shap_values = self.explainer.shap_values(X_instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            data=X_instance.iloc[0].values,
            feature_names=self.feature_names,
        )

        if show:
            shap.plots.waterfall(explanation, max_display=max_display)

        return explanation

    def plot_summary(
        self,
        X_data: pd.DataFrame,
        max_samples: int = 1000,
        plot_type: str = 'dot',
    ) -> Any:
        """Generate SHAP summary plot."""
        if not SHAP_AVAILABLE:
            raise SHAPExplainerError("SHAP not available")

        shap_values = self.explain_batch(X_data, max_samples)

        if plot_type == 'dot':
            shap.summary_plot(shap_values, X_data[self.feature_names], plot_type='dot', show=True)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X_data[self.feature_names], plot_type='bar', show=True)

        return shap_values


def compute_shap_stability(
    importance_history: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """Compute stability metrics for top-k features across time windows."""
    if importance_history.empty:
        return pd.DataFrame()

    overall_importance = importance_history.groupby('feature')['importance'].mean()
    top_features = overall_importance.nlargest(top_k).index.tolist()

    stability_records = []
    for feat in top_features:
        feat_data = importance_history[importance_history['feature'] == feat].sort_values('timestamp')
        if len(feat_data) >= 2:
            importance_series = feat_data['importance'].values
            cv = np.std(importance_series) / np.mean(importance_series) if np.mean(importance_series) > 0 else np.inf
            trend = np.corrcoef(range(len(importance_series)), importance_series)[0, 1] if len(importance_series) > 1 else 0

            stability_records.append({
                'feature': feat,
                'mean_importance': np.mean(importance_series),
                'cv': cv,
                'trend': trend,
                'n_windows': len(feat_data),
            })

    return pd.DataFrame(stability_records).sort_values('cv')


if __name__ == '__main__':
    if not SHAP_AVAILABLE:
        print("SHAP not installed. Run: pip install shap")
    else:
        print("SHAP explainer module ready")
        print(f"SHAP version: {shap.__version__}")