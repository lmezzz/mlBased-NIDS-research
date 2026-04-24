import pytest
from src.loader import load_KDD_TEST, load_KDD_TRAIN, load_CICI_COMBINED
from src.cleaner import clean_kdd, clean_cicids

def test_clean_KDD_TRAIN():
    df = load_KDD_TRAIN()
    cleaned = clean_kdd(df)
    assert cleaned is not None
    assert cleaned.shape[0] > 0
    assert cleaned.duplicated().sum() == 0
    assert cleaned.isnull().sum().sum() == 0
    assert "class" in cleaned.columns
    print(f"KDD TRAIN cleaned: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print(cleaned.head(3))

def test_clean_KDD_TEST():
    df = load_KDD_TEST()
    cleaned = clean_kdd(df)
    assert cleaned is not None
    assert cleaned.shape[0] > 0
    assert cleaned.duplicated().sum() == 0
    assert cleaned.isnull().sum().sum() == 0
    assert "class" in cleaned.columns
    print(f"KDD TEST cleaned: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print(cleaned.head(3))

def test_clean_CICIDS():
    df = load_CICI_COMBINED()
    cleaned = clean_cicids(df)
    assert cleaned is not None
    assert cleaned.shape[0] > 0
    assert cleaned.duplicated().sum() == 0
    assert cleaned.isnull().sum().sum() == 0
    assert "Label" in cleaned.columns
    print(f"CICIDS cleaned: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print(cleaned.head(3))
