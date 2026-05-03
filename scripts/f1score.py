import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

PLOT_DIR = Path("results/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/all_results.csv")

for model in df["model"].unique():
    model_df = df[df["model"] == model].sort_values("experiment")

    plt.figure(figsize=(10,6))

    plt.plot(model_df["experiment"], model_df["within_f1"],
             marker="o", label="Within F1")

    plt.plot(model_df["experiment"], model_df["cross_f1"],
             marker="o", label="Cross F1")

    plt.title(f"{model} - Within vs Cross Dataset Performance")
    plt.xlabel("Experiment")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.savefig(PLOT_DIR / f"{model}_within_vs_cross_f1.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    