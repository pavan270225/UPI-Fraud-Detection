"""
Drift detection module for UPI Fraud Detection pipeline.

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
for label-free drift monitoring on features and model scores.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DriftDetectionError(Exception):
    """Raised when drift detection fails."""
    pass


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    bin_method: str = 'quantile',
) -> Tuple[float, Dict]:
    """
    Calculate Population Stability Index (PSI) between reference and current distributions.

    Args:
        reference: Reference (training) distribution
        current: Current (monitoring) distribution
        bins: Number of bins
        bin_method: 'quantile' or 'uniform'

    Returns:
        (psi_value, details_dict)
    """
    # Remove NaN/inf
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan, {'error': 'Empty arrays after NaN removal'}

    # Create bins
    if bin_method == 'quantile':
        # Use reference quantiles for bin edges
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.unique(np.quantile(reference, quantiles))
        if len(bin_edges) < 2:
            return np.nan, {'error': 'Insufficient unique values for quantile binning'}
    else:
        # Uniform bins based on min/max of both
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Ensure bin edges cover all values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate proportions
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    # Avoid division by zero
    epsilon = 1e-6
    ref_props = np.maximum(ref_props, epsilon)
    cur_props = np.maximum(cur_props, epsilon)

    # PSI formula: sum((cur - ref) * ln(cur/ref))
    psi_values = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi = np.sum(psi_values)

    details = {
        'psi': psi,
        'bins': len(bin_edges) - 1,
        'bin_edges': bin_edges.tolist(),
        'ref_props': ref_props.tolist(),
        'cur_props': cur_props.tolist(),
        'psi_per_bin': psi_values.tolist(),
    }

    return psi, details


def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float, Dict]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Args:
        reference: Reference distribution
        current: Current distribution

    Returns:
        (ks_statistic, p_value, details_dict)
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan, np.nan, {'error': 'Empty arrays'}

    ks_stat, p_value = stats.ks_2samp(reference, current, alternative='two-sided', mode='auto')

    details = {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'ref_size': len(reference),
        'cur_size': len(current),
        'ref_mean': float(np.mean(reference)),
        'cur_mean': float(np.mean(current)),
        'ref_std': float(np.std(reference)),
        'cur_std': float(np.std(current)),
    }

    return ks_stat, p_value, details


def psi_on_dataframe(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bins: int = 10,
    bin_method: str = 'quantile',
) -> Dict[str, Dict]:
    """
    Calculate PSI for multiple columns.

    Args:
        reference_df: Reference dataframe
        current_df: Current dataframe
        columns: Columns to check (default: all numeric)
        bins: Number of bins
        bin_method: 'quantile' or 'uniform'

    Returns:
        Dict mapping column -> {psi, details, alert}
    """
    if columns is None:
        columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    results = {}
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            results[col] = {'error': 'Column not in both dataframes'}
            continue

        try:
            psi_val, details = population_stability_index(
                reference_df[col].values,
                current_df[col].values,
                bins=bins,
                bin_method=bin_method,
            )
            results[col] = {
                'psi': psi_val,
                'details': details,
                'alert': psi_val > 0.2,  # Standard threshold
            }
        except Exception as e:
            results[col] = {'error': str(e)}

    return results


class DriftMonitor:
    """
    Drift monitoring for features and model scores.

    Usage:
        monitor = DriftMonitor(psi_threshold=0.2, ks_threshold=0.05)
        monitor.fit(reference_data, feature_cols, score_col='risk_score')
        alerts = monitor.check(current_data)
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        bins: int = 10,
        bin_method: str = 'quantile',
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.bins = bins
        self.bin_method = bin_method

        self.reference_data = None
        self.feature_cols = None
        self.score_col = None
        self.fitted = False
        self.alerts_log = []

    def fit(
        self,
        reference_data: pd.DataFrame,
        feature_cols: List[str],
        score_col: Optional[str] = None,
    ) -> 'DriftMonitor':
        """
        Fit on reference (training) data.

        Args:
            reference_data: Training dataframe
            feature_cols: List of feature column names to monitor
            score_col: Optional model score column name
        """
        self.reference_data = reference_data.copy()
        self.feature_cols = feature_cols
        self.score_col = score_col
        self.fitted = True
        return self

    def check(
        self,
        current_data: pd.DataFrame,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check for drift on current data.

        Args:
            current_data: Current/monitoring dataframe
            timestamp: Optional timestamp for logging

        Returns:
            Dict with drift results and alerts
        """
        if not self.fitted:
            raise DriftDetectionError("DriftMonitor not fitted. Call fit() first.")

        if timestamp is None:
            timestamp = pd.Timestamp.now().isoformat()

        results = {
            'timestamp': timestamp,
            'features': {},
            'score': {},
            'alerts': [],
        }

        # Check feature drift (PSI)
        feature_results = psi_on_dataframe(
            self.reference_data[self.feature_cols],
            current_data[self.feature_cols],
            columns=self.feature_cols,
            bins=self.bins,
            bin_method=self.bin_method,
        )

        for col, res in feature_results.items():
            if 'error' in res:
                results['features'][col] = res
                continue

            alert = res['psi'] > self.psi_threshold
            results['features'][col] = {
                'psi': res['psi'],
                'threshold': self.psi_threshold,
                'alert': alert,
            }

            if alert:
                results['alerts'].append({
                    'type': 'feature_drift',
                    'column': col,
                    'metric': 'PSI',
                    'value': res['psi'],
                    'threshold': self.psi_threshold,
                    'timestamp': timestamp,
                })

        # Check score drift (PSI + KS)
        if self.score_col and self.score_col in current_data.columns and self.score_col in self.reference_data.columns:
            ref_scores = self.reference_data[self.score_col].values
            cur_scores = current_data[self.score_col].values

            psi_val, psi_details = population_stability_index(ref_scores, cur_scores, bins=self.bins, bin_method=self.bin_method)
            ks_stat, ks_p, ks_details = kolmogorov_smirnov_test(ref_scores, cur_scores)

            score_alert = (psi_val > self.psi_threshold) or (ks_stat > self.ks_threshold)

            results['score'] = {
                'psi': {'value': psi_val, 'threshold': self.psi_threshold, 'alert': psi_val > self.psi_threshold},
                'ks': {'value': ks_stat, 'p_value': ks_p, 'threshold': self.ks_threshold, 'alert': ks_stat > self.ks_threshold},
                'alert': score_alert,
            }

            if score_alert:
                alert_type = 'score_drift_psi' if psi_val > self.psi_threshold else 'score_drift_ks'
                results['alerts'].append({
                    'type': alert_type,
                    'column': self.score_col,
                    'metric': 'PSI' if psi_val > self.psi_threshold else 'KS',
                    'value': psi_val if psi_val > self.psi_threshold else ks_stat,
                    'threshold': self.psi_threshold if psi_val > self.psi_threshold else self.ks_threshold,
                    'timestamp': timestamp,
                })

        # Log alerts
        for alert in results['alerts']:
            self.alerts_log.append(alert)

        return results

    def get_alerts_log(self) -> pd.DataFrame:
        """Get alerts log as DataFrame."""
        if not self.alerts_log:
            return pd.DataFrame(columns=['type', 'column', 'metric', 'value', 'threshold', 'timestamp'])
        return pd.DataFrame(self.alerts_log)

    def save_alerts(self, path: Path) -> None:
        """Save alerts log to CSV."""
        df = self.get_alerts_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def save_config(self, path: Path) -> None:
        """Save monitor configuration."""
        config = {
            'psi_threshold': self.psi_threshold,
            'ks_threshold': self.ks_threshold,
            'bins': self.bins,
            'bin_method': self.bin_method,
            'feature_cols': self.feature_cols,
            'score_col': self.score_col,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


def detect_drift_rolling(
    data: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    window_size: int = 7,  # days
    step_size: int = 1,    # days
    psi_threshold: float = 0.2,
    score_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rolling window drift detection over time.

    Args:
        data: Full time-ordered dataframe
        feature_cols: Features to monitor
        time_col: Time column name
        window_size: Reference window in days
        step_size: Step between windows in days
        psi_threshold: PSI alert threshold
        score_col: Optional score column

    Returns:
        DataFrame with drift results per window
    """
    # Convert time to days
    time_days = (data[time_col] // 86400).astype(int) if data[time_col].max() > 1000 else data[time_col]
    min_day, max_day = time_days.min(), time_days.max()

    results = []
    monitor = DriftMonitor(psi_threshold=psi_threshold)

    for start_day in range(min_day, max_day - window_size, step_size):
        end_day = start_day + window_size

        # Reference: first window
        if start_day == min_day:
            ref_mask = (time_days >= start_day) & (time_days < end_day)
            ref_data = data[ref_mask].copy()
            monitor.fit(ref_data, feature_cols, score_col)
            continue

        # Current: subsequent windows
        cur_mask = (time_days >= start_day) & (time_days < end_day)
        cur_data = data[cur_mask].copy()

        if len(cur_data) == 0:
            continue

        drift_result = monitor.check(cur_data, timestamp=f"day_{start_day}_{end_day}")
        drift_result['window_start'] = start_day
        drift_result['window_end'] = end_day
        drift_result['ref_size'] = len(ref_data)
        drift_result['cur_size'] = len(cur_data)
        results.append(drift_result)

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0.3, 1.2, 1000)  # Shifted distribution

    psi, details = population_stability_index(ref, cur)
    print(f"PSI: {psi:.4f}")

    ks_stat, p_val, _ = kolmogorov_smirnov_test(ref, cur)
    print(f"KS: {ks_stat:.4f}, p={p_val:.4f}")

    print("Drift detection module ready")