import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src.preprocessor import encode_protocol, log_transform, fit_scaler, apply_scaler

def test_encode_protocol():
    # Setup
    df = pd.DataFrame({
        "protocol": ["tcp", "udp", "icmp", "tcp"],
        "other_feat": [1, 2, 3, 4]
    })
    
    # Execute
    result_df = encode_protocol(df)
    
    # Assert
    assert "protocol" not in result_df.columns
    assert "protocol_tcp" in result_df.columns
    assert "protocol_udp" in result_df.columns
    assert "protocol_icmp" in result_df.columns
    assert result_df["protocol_tcp"].iloc[0] == 1
    assert result_df["protocol_udp"].iloc[0] == 0
    assert result_df["protocol_tcp"].iloc[1] == 0
    assert result_df["protocol_udp"].iloc[1] == 1

def test_log_transform():
    # Setup
    df = pd.DataFrame({
        "feat1": [0, 1, 10],
        "feat2": [100, 1000, 10000],
        "cat_feat": ["a", "b", "c"]
    })
    
    # Execute
    result_df = log_transform(df)
    
    # Assert
    # log1p(0) = 0, log1p(1) = ln(2)
    assert np.isclose(result_df["feat1"].iloc[0], np.log1p(0))
    assert np.isclose(result_df["feat1"].iloc[1], np.log1p(1))
    assert np.isclose(result_df["feat1"].iloc[2], np.log1p(10))
    assert result_df["cat_feat"].equals(df["cat_feat"]) # Should be unchanged

def test_scaling_pipeline():
    # Setup
    train_df = pd.DataFrame({
        "feat1": [10, 20, 30],
        "feat2": [1, 2, 3],
        "label": ["A", "B", "A"]
    })
    test_df = pd.DataFrame({
        "feat1": [15, 25],
        "feat2": [1.5, 2.5],
        "label": ["B", "A"]
    })
    
    # Execute
    scaler, numeric_cols = fit_scaler(train_df)
    
    assert list(numeric_cols) == ["feat1", "feat2"]
    assert isinstance(scaler, StandardScaler)
    
    train_scaled = apply_scaler(train_df, scaler, numeric_cols)
    test_scaled = apply_scaler(test_df, scaler, numeric_cols)
    
    # Assert
    # Scaled training data should have mean 0 and std 1 (approximately)
    assert np.isclose(train_scaled["feat1"].mean(), 0)
    assert np.isclose(train_scaled["feat1"].std(ddof=0), 1)
    
    # Test data should be scaled using the SAME parameters
    # feat1 mean was 20, std was sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2)/3) = sqrt(200/3) = 8.1649
    # For 15: (15-20)/8.1649 = -5/8.1649 = -0.61237
    expected_val = (15 - 20) / np.sqrt(200/3)
    assert np.isclose(test_scaled["feat1"].iloc[0], expected_val)
