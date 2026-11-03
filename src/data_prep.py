"""
Data preparation module for UPI Fraud Detection pipeline.

Handles loading, preprocessing, and time-ordered splitting of IEEE-CIS and PaySim datasets.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np


class DataQualityError(Exception):
    """Raised when data validation fails."""
    pass


# Column dtype specifications for memory efficiency and consistency
IEEE_CIS_DTYPES = {
    'TransactionID': 'int64',
    'isFraud': 'int8',
    'TransactionDT': 'int64',
    'TransactionAmt': 'float32',
    'ProductCD': 'category',
    'card1': 'int32',
    'card2': 'float32',
    'card3': 'float32',
    'card4': 'category',
    'card5': 'float32',
    'card6': 'category',
    'addr1': 'float32',
    'addr2': 'float32',
    'dist1': 'float32',
    'dist2': 'float32',
    'P_emaildomain': 'category',
    'R_emaildomain': 'category',
}

PAYSIM_DTYPES = {
    'step': 'int32',
    'type': 'category',
    'amount': 'float32',
    'nameOrig': 'category',
    'oldbalanceOrg': 'float32',
    'newbalanceOrig': 'float32',
    'nameDest': 'category',
    'oldbalanceDest': 'float32',
    'newbalanceDest': 'float32',
    'isFraud': 'int8',
    'isFlaggedFraud': 'int8',
}


def load_ieee_cis(data_dir: Path, use_parquet: bool = True) -> pd.DataFrame:
    """
    Load IEEE-CIS Fraud Detection dataset.

    Args:
        data_dir: Directory containing train_transaction.csv or .parquet
        use_parquet: If True, try to load parquet first (faster)

    Returns:
        DataFrame with dtype enforcement
    """
    parquet_path = data_dir / 'train_transaction.parquet'
    csv_path = data_dir / 'train_transaction.csv'

    if use_parquet and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, dtype=IEEE_CIS_DTYPES, low_memory=False)
        df.to_parquet(parquet_path, index=False)
    else:
        raise FileNotFoundError(
            f"IEEE-CIS data not found in {data_dir}. "
            "Download from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data"
        )

    df = df.sort_values('TransactionDT').reset_index(drop=True)
    _validate_ieee_cis(df)
    return df


def load_paysim(data_dir: Path, use_parquet: bool = True) -> pd.DataFrame:
    """
    Load PaySim synthetic financial dataset.

    Args:
        data_dir: Directory containing paysim.csv or .parquet
        use_parquet: If True, try to load parquet first

    Returns:
        DataFrame with dtype enforcement
    """
    parquet_path = data_dir / 'paysim.parquet'
    csv_path = data_dir / 'paysim.csv'

    if use_parquet and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, dtype=PAYSIM_DTYPES, low_memory=False)
        df.to_parquet(parquet_path, index=False)
    else:
        raise FileNotFoundError(
            f"PaySim data not found in {data_dir}. "
            "Download from: https://github.com/albertogonzalez/paysim"
        )

    df = df.sort_values('step').reset_index(drop=True)
    _validate_paysim(df)
    return df


def _validate_ieee_cis(df: pd.DataFrame) -> None:
    """Validate IEEE-CIS dataframe integrity."""
    required_cols = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataQualityError(f"Missing required columns: {missing}")
    if not df['TransactionDT'].is_monotonic_increasing:
        raise DataQualityError("TransactionDT not monotonically increasing")
    fraud_rate = df['isFraud'].mean()
    if not 0.001 < fraud_rate < 0.1:
        raise DataQualityError(f"Unexpected fraud rate: {fraud_rate:.4f}")


def _validate_paysim(df: pd.DataFrame) -> None:
    """Validate PaySim dataframe integrity."""
    required_cols = ['step', 'type', 'amount', 'isFraud']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataQualityError(f"Missing required columns: {missing}")
    if not df['step'].is_monotonic_increasing:
        raise DataQualityError("step not monotonically increasing")


def time_ordered_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time order (no shuffling)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        fraud_rate = split_df['isFraud'].mean() if 'isFraud' in split_df.columns else 'N/A'
        print(f"  {name}: {len(split_df):,} rows ({len(split_df)/n*100:.1f}%), fraud_rate={fraud_rate}")
    return train_df, val_df, test_df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = 'isFraud',
    drop_cols: Optional[list] = None,
    require_target: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Separate features and target, drop non-feature columns."""
    default_drop = ['TransactionID', 'TransactionDT'] if 'TransactionID' in df.columns else []
    if drop_cols:
        default_drop.extend(drop_cols)
    drop_cols_final = [c for c in default_drop if c in df.columns]
    cols_to_drop = drop_cols_final.copy()
    if target_col in df.columns:
        cols_to_drop.append(target_col)
    X = df.drop(columns=cols_to_drop)

    if target_col in df.columns:
        y = df[target_col].astype('int8')
    elif require_target:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    else:
        y = None

    # Encode categorical columns to integer codes so sklearn models can consume them.
    # Mirrors dashboard/app.py prepare_xy behavior.
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes
    X = X.fillna(-999)
    return X, y  # type: ignore[return-value]


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'class_weight',
) -> Dict:
    """Return parameters for handling class imbalance."""
    fraud_rate = y_train.mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate
    if method == 'class_weight':
        return {'scale_pos_weight': scale_pos_weight, 'class_weight': 'balanced'}
    elif method == 'smote':
        return {'scale_pos_weight': 1.0}
    else:
        return {'scale_pos_weight': 1.0}


if __name__ == '__main__':
    import sys
    data_root = Path('/Users/pavanc/Downloads/UPI_Fraud_Detection_Project/data')
    print("Testing data loading...")
    try:
        df = load_ieee_cis(data_root / 'ieee-cis')
        print(f"IEEE-CIS loaded: {df.shape}")
    except FileNotFoundError as e:
        print(f"IEEE-CIS not found: {e}", file=sys.stderr)
    try:
        df = load_paysim(data_root / 'paysim')
        print(f"PaySim loaded: {df.shape}")
    except FileNotFoundError as e:
        print(f"PaySim not found: {e}", file=sys.stderr)
