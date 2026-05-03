# src/experiments/attack_breakdown.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from config import MODELS_DIR, CICIDS_CONTROL, CICIDS_LABELS, PROCESSED_DIR , CICIDS_EXP1 , CICIDS_EXP2 , CICIDS_EXP3 , CICIDS_EXP4 , CICIDS_EXP5
from src.preprocessor import apply_scaler



def attack_breakdown(model_name, exp_name, cicids_path):
    saved        = joblib.load(MODELS_DIR / f"{model_name}_{exp_name}.pkl")
    model        = saved["model"]
    scaler       = saved["scaler"]
    feature_cols = saved["features"]
    numeric_cols = saved["numeric_cols"]

    cicids = pd.read_csv(cicids_path)
    labels = pd.read_csv(CICIDS_LABELS)["Label"]

    assert len(cicids) == len(labels), \
        f"Row mismatch: cicids={len(cicids)}, labels={len(labels)}"

    # scale using only the numeric cols the scaler was fitted on
    cicids[numeric_cols] = scaler.transform(cicids[numeric_cols])

    y_pred = model.predict(cicids[feature_cols])

    results = []
    for attack_type in labels.unique():
        mask       = (labels == attack_type).values
        total      = mask.sum()
        caught     = (y_pred[mask] == 1).sum()
        missed     = (y_pred[mask] == 0).sum()
        catch_rate = round(caught / total, 4) if total > 0 else 0
        results.append({
            "attack_type": attack_type,
            "total":       total,
            "caught":      caught,
            "missed":      missed,
            "catch_rate":  catch_rate
        })

    df = pd.DataFrame(results).sort_values("catch_rate", ascending=True)
    print(f"\n[{model_name.upper()}] {exp_name.upper()} — Per-Attack Breakdown")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    attack_breakdown("lr",  "exp0", CICIDS_CONTROL)
    attack_breakdown("rf",  "exp0", CICIDS_CONTROL)
    attack_breakdown("svm", "exp0", CICIDS_CONTROL)
    attack_breakdown("lr",  "exp1", CICIDS_EXP1)
    attack_breakdown("rf",  "exp1", CICIDS_EXP1)
    attack_breakdown("svm", "exp1", CICIDS_EXP1)
    attack_breakdown("lr",  "exp2", CICIDS_EXP2)
    attack_breakdown("rf",  "exp2", CICIDS_EXP2)
    attack_breakdown("svm", "exp2", CICIDS_EXP2)    
    attack_breakdown("lr",  "exp3", CICIDS_EXP3)
    attack_breakdown("rf",  "exp3", CICIDS_EXP3)
    attack_breakdown("svm", "exp3", CICIDS_EXP3)
    attack_breakdown("lr",  "exp4", CICIDS_EXP4)
    attack_breakdown("rf",  "exp4", CICIDS_EXP4)
    attack_breakdown("svm", "exp4", CICIDS_EXP4)
    attack_breakdown("lr",  "exp5", CICIDS_EXP5)
    attack_breakdown("rf",  "exp5", CICIDS_EXP5)
    attack_breakdown("svm", "exp5", CICIDS_EXP5)
    
    