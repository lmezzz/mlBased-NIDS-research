import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. ONE-HOT ENCODING (protocol)
# -----------------------------
def encode_protocol(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["protocol"])  # it will create protocol_tcp, protocol_udp, protocol_icmp basically different columns for each protocol with 0/1 values
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

SERVICE_BUCKET_CATEGORIES = [
    "http", "https", "ssh", "ftp", "dns", "smtp",
    "telnet", "mysql", "rdp", "system", "user", "dynamic", "unknown"
]

def encode_service_bucket(df):
    df = df.copy()
    if "service_bucket" not in df.columns:
        return df
    # use pd.Categorical to ensure consistent columns across datasets
    df["service_bucket"] = pd.Categorical(
        df["service_bucket"],
        categories=SERVICE_BUCKET_CATEGORIES
    )
    dummies = pd.get_dummies(df["service_bucket"], prefix="svc")
    bool_cols = dummies.select_dtypes(include="bool").columns
    dummies[bool_cols] = dummies[bool_cols].astype(int)
    df = df.drop("service_bucket", axis=1)
    df = pd.concat([df, dummies], axis=1)
    return df

#fixing the skewness of the data by applying log transformation to numeric features
RATIO_COLS = {
    "syn_ratio", "rst_ratio", "fin_ratio",
    "data_pkt_ratio", "window_ratio", "byte_ratio",
    "label"
}

def log_transform(df):
    df = df.copy()
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in RATIO_COLS
    ]
    for col in numeric_cols:
        df[col] = np.log1p(df[col])
    return df

#to keep the same scale of the features, we will apply standardization to the numeric features
#if we apply scaling to the full data set then we will have data leakage, so we will fit the scaler on the training data and then apply the same scaler to the test data

def fit_scaler(train_df):
    numeric_cols = train_df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler, numeric_cols

def apply_scaler(df, scaler, numeric_cols):
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df