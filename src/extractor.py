import numpy as np
import pandas as pd

def feature_control_kdd(df: pd.DataFrame) -> pd.DataFrame:
    features = [
    "duration",
    "protocol",
    "src_bytes",
    "dst_bytes",
    "bytes_per_sec",
    "byte_ratio"  
    ]

    df = df.copy()

    # Rename columns to unified schema
    df = df.rename(columns={
        "protocol_type": "protocol"
    })

    # Ensure numeric
    df["duration"] = df["duration"].astype(float)
    df["src_bytes"] = df["src_bytes"].astype(float)
    df["dst_bytes"] = df["dst_bytes"].astype(float)


    # Feature engineering
    df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1e-6)

    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

    print(f"[Extractor] KDD feature control applied. Resulting shape: {df.shape}")
    print(f"[Extractor] Sample features:\n{df[features].head(3)}")

    return df[[
        "duration",
        "protocol",
        "src_bytes",
        "dst_bytes",
        "bytes_per_sec",
        "byte_ratio"
    ]]

def feature_control_cicids(df: pd.DataFrame) -> pd.DataFrame:
    features = [
    "duration",
    "protocol",
    "src_bytes",
    "dst_bytes",
    "bytes_per_sec",
    "byte_ratio"
    ]

    df = df.copy()

    # Rename to match schema
    df = df.rename(columns={
        "Flow Duration": "duration",
        "Protocol": "protocol",
        "Total Length of Fwd Packets": "src_bytes",
        "Total Length of Bwd Packets": "dst_bytes"
    })

    # Convert duration (CICIDS is usually in microseconds)
    df["duration"] = df["duration"] / 1e6

    # Ensure numeric
    df["src_bytes"] = df["src_bytes"].astype(float)
    df["dst_bytes"] = df["dst_bytes"].astype(float)

    #converting the protocol int to string to make it same as KDD
    protocol_map = {6: "tcp", 17: "udp", 1: "icmp"}
    df["protocol"] = df["protocol"].map(protocol_map)

    # Feature engineering
    df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1e-6)
    
    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

    print(f"[Extractor] CICIDS feature control applied. Resulting shape: {df.shape}")
    print(f"[Extractor] Sample features:\n{df[features].head(3)}")

    return df[[
        "duration",
        "protocol",
        "src_bytes",
        "dst_bytes",
        "bytes_per_sec",
        "byte_ratio"
    ]]