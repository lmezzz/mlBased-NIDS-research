import pandas as pd
import os
import glob

all_results = []

for filepath in glob.glob("results/feature_importance/*.csv"):
    df = pd.read_csv(filepath)
    # extract model and experiment from filename
    # e.g. "lr_importance_exp0.csv" → model=lr, experiment=exp0
    filename = os.path.basename(filepath).replace(".csv", "")
    parts = filename.split("_")
    df["model"]      = parts[0]        # lr, rf, svm
    df["experiment"] = parts[-1]       # exp0, exp1...
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("results/feature_importance_all.csv", index=False)
print(f"Saved → results/feature_importance_all.csv {combined.shape}")