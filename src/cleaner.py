import pandas as pd
import numpy as np

# ── NSL-KDD label column is called "class"
# ── CICIDS label column is called "Label" (already stripped in combined file)

def clean_kdd(df: pd.DataFrame) -> pd.DataFrame:
    print("[Cleaner] Cleaning KDD dataset...")
    print(f"[Cleaner] Shape before: {df.shape}")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[Cleaner] Duplicates removed: {before - len(df)}")

    # Drop nulls
    before = len(df)
    df.dropna(inplace=True)
    print(f"[Cleaner] Nulls removed: {before - len(df)}")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Confirm label column exists
    assert "class" in df.columns, "[Cleaner] ERROR: 'class' column not found in KDD"

    print(f"[Cleaner] Shape after: {df.shape}")
    print(f"[Cleaner] Label distribution:\n{df['class'].value_counts()}")
    return df


def clean_cicids(df: pd.DataFrame) -> pd.DataFrame:
    print("[Cleaner] Cleaning CICIDS dataset...")
    print(f"[Cleaner] Shape before: {df.shape}")

    # Strip column name spaces (safety check)
    df.columns = df.columns.str.strip()

    # Replace infinity values with NaN then drop
    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    print(f"[Cleaner] Infinity values found: {inf_count}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop nulls
    before = len(df)
    df.dropna(inplace=True)
    print(f"[Cleaner] Nulls removed: {before - len(df)}")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[Cleaner] Duplicates removed: {before - len(df)}")

    # Confirm label column exists
    assert "Label" in df.columns, "[Cleaner] ERROR: 'Label' column not found in CICIDS"

    print(f"[Cleaner] Shape after: {df.shape}")
    print(f"[Cleaner] Label distribution:\n{df['Label'].value_counts()}")
    return df