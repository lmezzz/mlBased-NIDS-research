import pandas as pd

KDD_LABEL_COL   = "class"
CICI_LABEL_COL  = "Label"
FEATURE_COLS    = ["duration", "protocol", "src_bytes",
                   "dst_bytes", "bytes_per_sec", "byte_ratio"]

def align_kdd(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning KDD...")
    out = df[FEATURE_COLS].copy()
    out["label"] = (df[KDD_LABEL_COL] != "normal").astype(int)
    print(f"[Aligner] label distribution:\n{out['label'].value_counts()}")
    return out

def align_cicids(df: pd.DataFrame) -> pd.DataFrame:
    print("[Aligner] Aligning CICIDS...")
    out = df[FEATURE_COLS].copy()
    out["label"] = (df[CICI_LABEL_COL] != "BENIGN").astype(int)
    print(f"[Aligner] label distribution:\n{out['label'].value_counts()}")
    return out