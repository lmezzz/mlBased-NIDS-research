import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# ── IMPORT CONFIG ──
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CICI_RAW_PATH, PROCESSED_DIR

# ── FILE NAMES ──
all_files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# ── LOAD ALL FILES ──
dfs = []
for file in all_files:
    # skip lock files
    if file.startswith(".~lock"):
        print(f"⚠ Skipping lock file: {file}")
        continue

    path = CICI_RAW_PATH / file
    if path.exists():
        try:
            temp = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            temp = pd.read_csv(path, encoding="latin1", low_memory=False)
        dfs.append(temp)
        print(f"✓ Loaded: {file} → {temp.shape}")
    else:
        print(f"✗ NOT FOUND: {path}")

# ── COMBINE ──
print("\nCombining all files...")
cicids_df = pd.concat(dfs, ignore_index=True)
print("Combined shape:", cicids_df.shape)

# ── CLEAN COLUMN NAMES ──
cicids_df.columns = cicids_df.columns.str.strip()

# ── HANDLE INF AND NAN ──
cicids_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cicids_df.dropna(inplace=True)
print("\nShape after dropping NaN/inf:", cicids_df.shape)

# ── CHECK LABELS BEFORE SAMPLING ──
print("\nAll labels before sampling:")
print(cicids_df["Label"].value_counts())

# ── SMART SAMPLING WITH MINIMUM THRESHOLD ──
MIN_ROWS = 10      # minimum rows we want per class
SAMPLE_RATE = 0.10 # take 10% from classes that have enough rows

print(f"\nSampling with minimum {MIN_ROWS} rows per class...")

sampled_parts = []
label_counts = cicids_df["Label"].value_counts()

for label, count in label_counts.items():
    label_df = cicids_df[cicids_df["Label"] == label]
    sampled_count = int(count * SAMPLE_RATE)

    if sampled_count < MIN_ROWS:
        # not enough after sampling — take all rows for this class
        sampled_parts.append(label_df)
        print(f"  {label}: only {count} rows total → keeping all {count}")
    else:
        # enough rows — sample normally
        sampled = label_df.sample(frac=SAMPLE_RATE, random_state=42)
        sampled_parts.append(sampled)
        print(f"  {label}: {count} rows → sampled {len(sampled)}")

# ── COMBINE SAMPLED PARTS ──
cicids_small = pd.concat(sampled_parts, ignore_index=True)

# ── SHUFFLE (so classes aren't grouped together) ──
cicids_small = cicids_small.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal shape:", cicids_small.shape)
print("\nFinal label distribution:")
print(cicids_small["Label"].value_counts())

# ── SAVE ──
output_path = PROCESSED_DIR / "cicids_combined.csv"
cicids_small.to_csv(output_path, index=False)
print(f"\nSaved to {output_path} ✓")