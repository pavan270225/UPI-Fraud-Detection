"""Unit tests for data_prep module — categorical encoding + validation."""

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import pytest

from src.data_prep import (
    DataQualityError,
    handle_class_imbalance,
    load_ieee_cis,
    load_paysim,
    prepare_features_target,
    time_ordered_split,
)


def test_prepare_features_target_encodes_categories():
    """Categorical columns must be encoded to int codes for sklearn compatibility."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [100, 200, 300, 400],
        "type": pd.Categorical(["PAYMENT", "TRANSFER", "PAYMENT", "CASH_OUT"]),
        "amount": [100.0, 200.0, 300.0, 400.0],
        "isFraud": [0, 1, 0, 1],
    })
    X, y = prepare_features_target(df, target_col="isFraud")
    # type column should be numeric codes, not strings/categories
    assert not is_categorical_dtype(X["type"])
    assert X["type"].dtype in (np.int64, np.int32, np.int8)
    assert set(X["type"].unique()) == {0, 1, 2}  # 3 categories -> codes 0,1,2
    assert list(y) == [0, 1, 0, 1]


def test_prepare_features_target_drops_id_and_time():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [100, 200],
        "amount": [100.0, 200.0],
        "isFraud": [0, 1],
    })
    X, y = prepare_features_target(df, target_col="isFraud")
    assert "TransactionID" not in X.columns
    assert "TransactionDT" not in X.columns
    assert "amount" in X.columns


def test_time_ordered_split_no_shuffle():
    df = pd.DataFrame({
        "TransactionDT": list(range(100)),
        "isFraud": [0] * 100,
    })
    train, val, test = time_ordered_split(df, "TransactionDT", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    # Order preserved
    assert train["TransactionDT"].iloc[0] == 0
    assert test["TransactionDT"].iloc[-1] == 99


def test_handle_class_imbalance_scale_pos_weight():
    y = pd.Series([0, 0, 0, 1])  # fraud_rate = 0.25
    params = handle_class_imbalance(None, y, method="class_weight")
    expected = (1 - 0.25) / 0.25  # = 3.0
    assert params["scale_pos_weight"] == pytest.approx(expected)
    assert params["class_weight"] == "balanced"


def test_load_ieee_cis_missing_file(tmp_path):
    empty_dir = tmp_path / "no_data"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_ieee_cis(empty_dir)


def test_load_paysim_missing_file(tmp_path):
    empty_dir = tmp_path / "no_data"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_paysim(empty_dir)
