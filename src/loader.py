from scipy.io import arff
import pandas as pd
from config import CICIDS_CONTROL, KDD_RAW_TEST_PATH, KDD_RAW_TRAIN_PATH, CICI_COMBINED_RAW_PATH, KDD_TEST_CONTROL, KDD_TRAIN_CONTROL

def load_KDD_TEST() -> pd.DataFrame:
    print('[Loader] Loading the KDD test dataset')
    data , meta = arff.loadarff(KDD_RAW_TEST_PATH)
    df = pd.DataFrame(data)

    df = df.apply(lambda col: col.map(
    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    ))
    print(f"[Loader] KDD test dataset loaded with rows: {df.shape[0]}, columns: {df.shape[1]}")

    return df

def load_KDD_TRAIN() -> pd.DataFrame:
    print('[Loader] Loading the KDD train dataset')
    data , meta = arff.loadarff(KDD_RAW_TRAIN_PATH)
    df = pd.DataFrame(data)
    df = df.apply(lambda col: col.map(
    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    ))
    print(f"[Loader] KDD train dataset loaded with rows: {df.shape[0]}, columns: {df.shape[1]}")
    return df

def load_CICI_COMBINED() -> pd.DataFrame:
    print('[Loader] Loading the CICI combined dataset')
    df = pd.read_csv(CICI_COMBINED_RAW_PATH)
    return df


def load_CICI_CONTROL() -> pd.DataFrame:
    print('[Loader] Loading the CICI control dataset')
    df = pd.read_csv(CICIDS_CONTROL)
    return df

def load_KDD_CONTROL() -> pd.DataFrame:
    print('[Loader] Loading the KDD control dataset')
    df = pd.read_csv(KDD_TRAIN_CONTROL)
    return df

def load_KDD_TEST_CONTROL() -> pd.DataFrame:
    print('[Loader] Loading the KDD test control dataset')
    df = pd.read_csv(KDD_TEST_CONTROL)
    return df




