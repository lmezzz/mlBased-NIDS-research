import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.loader import load_CICI_FRI_AFT

df = load_CICI_FRI_AFT()
print("Shape:", df.shape)
print("\nAll columns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nUnique labels:")
print(df[" Label"].unique())