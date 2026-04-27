import pandas as pd

CONTROL_FEATURE_COLS    = ["duration", "src_bytes",
                   "dst_bytes", "bytes_per_sec", "byte_ratio" , "protocol_icmp" , "protocol_tcp", "protocol_udp"]

def align_kdd(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning KDD...")
    out = df[CONTROL_FEATURE_COLS].copy()
    out["label"] = (df["label"] != "normal").astype(int)
    print(f"[Aligner] label distribution:\n{out['label'].value_counts()}")
    return out

def align_cicids(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning CICIDS...")
    out = df[CONTROL_FEATURE_COLS].copy()
    out["label"] = (df["label"] != "BENIGN").astype(int)
    print(f"[Aligner] label distribution:\n{out['label'].value_counts()}")
    return out

PROTOCOL_AWARE_COLS = [
        "syn_ratio", "rst_ratio", "fin_ratio",
        "data_pkt_ratio", "service_bucket"
    ]

def align_protocol_aware_kdd(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning KDD with protocol-aware features...")
    out = df[PROTOCOL_AWARE_COLS].copy()
    print(f"[Aligner] Sample protocol-aware features:\n{out[PROTOCOL_AWARE_COLS].head(3)}")
    return out

def align_protocol_aware_cicids(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning CICIDS with protocol-aware features...")
    out = df[PROTOCOL_AWARE_COLS].copy()
    print(f"[Aligner] Sample protocol-aware features:\n{out[PROTOCOL_AWARE_COLS].head(3)}")
    return out

