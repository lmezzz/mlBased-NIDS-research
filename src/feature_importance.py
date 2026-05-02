import joblib
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, RESULTS_DIR
def analyse_lr(exp_name):
    saved = joblib.load(MODELS_DIR / f"lr_{exp_name}.pkl")
    model    = saved["model"]
    features = saved["features"]
    
    coef_df = pd.DataFrame({
        "feature":     features,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    
    print(f"\n[LR] Feature importance — {exp_name.upper()}")
    print(coef_df.to_string(index=False))

    coef_df.to_csv(
        RESULTS_DIR / f"lr_importance_{exp_name}.csv",
        index=False
    )
    print(f"[LR] Importance saved → lr_importance_{exp_name}.csv")

    return coef_df


def analyse_rf(exp_name):
    saved = joblib.load(MODELS_DIR / f"rf_{exp_name}.pkl")
    model    = saved["model"]
    features = saved["features"]
    
    imp_df = pd.DataFrame({
        "feature":    features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\n[RF] Feature importance — {exp_name.upper()}")
    print(imp_df.to_string(index=False))

    imp_df.to_csv(
        RESULTS_DIR / f"rf_importance_{exp_name}.csv",
        index=False
    )
    print(f"[RF] Importance saved → rf_importance_{exp_name}.csv")
    return imp_df


def analyse_svm(exp_name):
    saved = joblib.load(MODELS_DIR / f"svm_{exp_name}.pkl")
    model    = saved["model"]
    features = saved["features"]
    
    coef_df = pd.DataFrame({
        "feature":     features,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    
    print(f"\n[SVM] Feature importance — {exp_name.upper()}")
    print(coef_df.to_string(index=False))
    coef_df.to_csv(
        RESULTS_DIR / f"svm_importance_{exp_name}.csv",
        index=False
    )
    print(f"[SVM] Importance saved → svm_importance_{exp_name}.csv")
    return coef_df


def run_all():
    for exp in ["exp0", "exp1", "exp2", "exp3", "exp4", "exp5"]:
        analyse_lr(exp)
        analyse_rf(exp)
        analyse_svm(exp)


if __name__ == "__main__":
    run_all()