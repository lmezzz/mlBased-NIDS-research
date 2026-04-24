import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. ONE-HOT ENCODING (protocol)
# -----------------------------
def encode_protocol(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["protocol"])  # it will create protocol_tcp, protocol_udp, protocol_icmp basically different columns for each protocol with 0/1 values
    return df


#fixing the skewness of the data by applying log transformation to numeric features
def log_transform(df):
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=np.number).columns 
    
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