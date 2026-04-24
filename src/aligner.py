import pandas as pd

FEATURE_COLS    = ["duration", "src_bytes",
                   "dst_bytes", "bytes_per_sec", "byte_ratio" , "protocol_icmp" , "protocol_tcp", "protocol_udp"]

def align_kdd(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning KDD...")
    out = df[FEATURE_COLS].copy()
    out["label"] = (df["label"] != "normal").astype(int)
    print(f"[Aligner] label distribution:\n{out['label'].value_counts()}")
    return out

def align_cicids(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning CICIDS...")
    out = df[FEATURE_COLS].copy()
    out["label"] = (df["label"] != "BENIGN").astype(int)
    print(f"[Aligner] label distribution:\n{out['label'].value_counts()}")
    return out