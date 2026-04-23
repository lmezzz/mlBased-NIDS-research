import pytest
from src.loader import load_CICI_FRI_AFT, load_KDD_TEST, load_KDD_TRAIN

def test_load_KDD_TEST():
    df = load_KDD_TEST()
    assert df is not None
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    print(f"KDD TEST dataset loaded with rows {df.shape[0]} and columns {df.shape[1]}")
    print(df.head(3))

def test_load_KDD_TRAIN():
    df = load_KDD_TRAIN()
    assert df is not None 
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    print(f"KDD TRAIN dataset loaded with rows {df.shape[0]} and columns {df.shape[1]}")
    print(df.head(3))

def test_load_CICI_FRI_AFT():
    df = load_CICI_FRI_AFT()
    assert df is not None 
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    print(f"CICI Friday Afternoon dataset loaded with rows {df.shape[0]} and columns {df.shape[1]}")
    print(df.head(3))