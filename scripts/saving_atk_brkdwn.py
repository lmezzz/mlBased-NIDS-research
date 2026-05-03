import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, CICIDS_CONTROL, CICIDS_LABELS, PROCESSED_DIR , CICIDS_EXP1 , CICIDS_EXP2 , CICIDS_EXP3 , CICIDS_EXP4 , CICIDS_EXP5
import pandas as pd
from scripts.attack_breakdown import attack_breakdown

if __name__ == "__main__":
    from config import (
        CICIDS_CONTROL, CICIDS_EXP1, CICIDS_EXP2,
        CICIDS_EXP3, CICIDS_EXP4, CICIDS_EXP5
    )

    exp_paths = {
        "exp0": CICIDS_CONTROL,
        "exp1": CICIDS_EXP1,
        "exp2": CICIDS_EXP2,
        "exp3": CICIDS_EXP3,
        "exp4": CICIDS_EXP4,
        "exp5": CICIDS_EXP5,
    }

    all_results = []
    for model_name in ["lr", "rf", "svm"]:
        for exp_name, cicids_path in exp_paths.items():
            df = attack_breakdown(model_name, exp_name, cicids_path)
            df["model"]      = model_name
            df["experiment"] = exp_name
            all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    os.makedirs("results", exist_ok=True)
    combined.to_csv("results/attack_breakdown_all.csv", index=False)
    print(f"\nSaved → results/attack_breakdown_all.csv")
    print(f"Shape: {combined.shape}")