import numpy as np
import pandas as pd

from src.loader import load_KDD_TRAIN , load_KDD_TEST, load_CICI_COMBINED
from src.cleaner import clean_kdd , clean_cicids
from src.extractor import feature_control_kdd , feature_control_cicids


def test_feature_control_kdd():
    # -----------------------------
    # Setup
    # -----------------------------
    df = load_KDD_TRAIN()
    df = clean_kdd(df)
    df = df.head(10).copy()

    # ensure numeric safety (optional but good practice)
    df["src_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce")
    df["dst_bytes"] = pd.to_numeric(df["dst_bytes"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # -----------------------------
    # Execute
    # -----------------------------

    result_df = feature_control_kdd(df)

    # -----------------------------
    # 1. STRUCTURE CHECK
    # -----------------------------
    assert "bytes_per_sec" in result_df.columns
    assert "byte_ratio" in result_df.columns

    # -----------------------------
    # 2. NO NaN / INF CHECK (IMPORTANT)
    # -----------------------------
    assert not result_df.isna().any().any(), "NaN values found in features"
    assert np.isfinite(
        result_df.select_dtypes(include=np.number)
    ).all().all(), "Inf values found in features"

    # -----------------------------

def test_feature_control_cicids():
    # -----------------------------
    # Setup
    # -----------------------------
    df = load_CICI_COMBINED()
    df = clean_cicids(df)
    df = df.head(10).copy()

    # ensure numeric safety (optional but good practice)
    df["Total Length of Fwd Packets"] = pd.to_numeric(df["Total Length of Fwd Packets"], errors="coerce")
    df["Total Length of Bwd Packets"] = pd.to_numeric(df["Total Length of Bwd Packets"], errors="coerce")
    df["Flow Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

    # -----------------------------
    # Execute
    # -----------------------------

    result_df = feature_control_cicids(df)

    # -----------------------------
    # 1. STRUCTURE CHECK
    # -----------------------------
    assert "bytes_per_sec" in result_df.columns
    assert "byte_ratio" in result_df.columns

    # -----------------------------
    # 2. NO NaN / INF CHECK (IMPORTANT)
    # -----------------------------
    assert not result_df.isna().any().any(), "NaN values found in features"
    assert np.isfinite(
        result_df.select_dtypes(include=np.number)
    ).all().all(), "Inf values found in features"