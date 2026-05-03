"""
visualizations.py
Generates all research paper charts from results CSVs.
Run from project root: python src/experiments/visualizations.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── STYLE ──
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

COLORS = {"lr": "#2196F3", "rf": "#4CAF50", "svm": "#FF5722"}

MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest", "svm": "SVM"}

EXP_LABELS = {
    "exp0": "EXP0\nControl",
    "exp1": "EXP1\n+syn_ratio",
    "exp2": "EXP2\n+rst_ratio",
    "exp3": "EXP3\n+fin_ratio",
    "exp4": "EXP4\n+data_pkt",
    "exp5": "EXP5\n+service",
}

EXPS   = ["exp0", "exp1", "exp2", "exp3", "exp4", "exp5"]
MODELS = ["lr", "rf", "svm"]

os.makedirs("results/figures", exist_ok=True)

# ── LOAD DATA ──
# ── LOAD + NORMALIZE ──
results    = pd.read_csv("results/experiment_results.csv")
importance = pd.read_csv("results/feature_importance_all.csv")
breakdown  = pd.read_csv("results/attack_breakdown_all.csv")

# normalize results to match other CSVs
results["model"]      = results["model"].str.lower().str.replace("logisticregression", "lr").str.replace("randomforest", "rf").str.replace("svm", "svm")
results["experiment"] = results["experiment"].str.lower()




# ══════════════════════════════════════════════════════
# CHART 1 — Cross-Dataset F1 Across Experiments
# Shows how generalization degrades as protocol features are added
# KEY FINDING: every protocol feature added makes cross F1 worse
# ══════════════════════════════════════════════════════
def chart1_cross_f1_trend():
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in MODELS:
        df = results[results["model"] == model].sort_values("experiment")
        ax.plot(
            [EXP_LABELS[e] for e in EXPS],
            [df[df["experiment"] == e]["cross_f1"].values[0] for e in EXPS],
            marker="o", linewidth=2.5, markersize=7,
            color=COLORS[model], label=MODEL_LABELS[model]
        )

    ax.set_title("Cross-Dataset F1 Score Across Experiments\n(Train: NSL-KDD → Test: CICIDS 2017)", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment (Feature Added)", fontsize=11)
    ax.set_ylabel("Cross-Dataset F1 Score", fontsize=11)
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=10)
    ax.axhline(y=0.3011, color="#FF5722", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(5.1, 0.305, "SVM EXP0\nbest: 0.301", fontsize=8, color="#FF5722", alpha=0.7)

    plt.tight_layout()
    plt.savefig("results/figures/chart1_cross_f1_trend.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart1_cross_f1_trend.png")
    print("  Significance: Shows that every protocol feature added degrades generalization.")
    print("  SVM collapses hardest at EXP2 (rst_ratio). Best performance is EXP0 for all models.")


# ══════════════════════════════════════════════════════
# CHART 2 — Within vs Cross F1 Gap (Grouped Bar)
# Shows the illusion of benchmark performance
# KEY FINDING: highest within-F1 models have worst cross-F1
# ══════════════════════════════════════════════════════
def chart2_within_vs_cross_gap():
    highlight_exps = ["exp0", "exp2", "exp4"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, exp in enumerate(highlight_exps):
        ax = axes[idx]
        x = np.arange(len(MODELS))
        width = 0.35

        within_vals = [results[(results["model"] == m) & (results["experiment"] == exp)]["within_f1"].values[0] for m in MODELS]
        cross_vals  = [results[(results["model"] == m) & (results["experiment"] == exp)]["cross_f1"].values[0]  for m in MODELS]

        bars1 = ax.bar(x - width/2, within_vals, width, label="Within-Dataset", 
                       color=[COLORS[m] for m in MODELS], alpha=0.9)
        bars2 = ax.bar(x + width/2, cross_vals,  width, label="Cross-Dataset",
                       color=[COLORS[m] for m in MODELS], alpha=0.4)

        ax.set_title(EXP_LABELS[exp].replace("\n", " "), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "RF", "SVM"])
        ax.set_ylim(0, 1.0)
        if idx == 0:
            ax.set_ylabel("F1 Score")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8, alpha=0.7)

    solid = mpatches.Patch(color="gray", alpha=0.9, label="Within-Dataset F1")
    faded = mpatches.Patch(color="gray", alpha=0.4, label="Cross-Dataset F1")
    fig.legend(handles=[solid, faded], loc="upper right", fontsize=10)
    fig.suptitle("Within-Dataset vs Cross-Dataset F1 — EXP0, EXP2, EXP4",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig("results/figures/chart2_within_vs_cross_gap.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart2_within_vs_cross_gap.png")
    print("  Significance: High within-F1 does not predict cross-F1.")
    print("  The benchmark gap is visible — models look good but fail in deployment.")


# ══════════════════════════════════════════════════════
# CHART 3 — Performance Drop Per Experiment
# Shows which feature caused the most damage
# KEY FINDING: rst_ratio causes largest drop, especially SVM
# ══════════════════════════════════════════════════════
def chart3_performance_drop():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EXPS))
    width = 0.25

    for i, model in enumerate(MODELS):
        drops = [results[(results["model"] == model) & 
                         (results["experiment"] == exp)]["performance_drop"].values[0] 
                 for exp in EXPS]
        ax.bar(x + i*width, drops, width, label=MODEL_LABELS[model],
               color=COLORS[model], alpha=0.85)

    ax.set_title("Performance Drop (Within F1 − Cross F1) Per Experiment",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("F1 Drop")
    ax.set_xticks(x + width)
    ax.set_xticklabels([EXP_LABELS[e].replace("\n", " ") for e in EXPS], fontsize=9)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("results/figures/chart3_performance_drop.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart3_performance_drop.png")
    print("  Significance: EXP2 (rst_ratio) causes the largest drop for all models.")
    print("  SVM shows the most dramatic damage — maximum margin commits hard to wrong boundary.")


# ══════════════════════════════════════════════════════
# CHART 4 — Attack Type Catch Rate Heatmap (EXP0)
# Shows which attack types each model catches vs misses
# KEY FINDING: modern HTTP-based attacks nearly always missed
# ══════════════════════════════════════════════════════
def chart4_attack_heatmap():
    exp0 = breakdown[breakdown["experiment"] == "exp0"]
    
    pivot = exp0.pivot_table(
        index="attack_type", columns="model", values="catch_rate"
    )
    pivot.columns = [MODEL_LABELS[c] for c in pivot.columns]
    pivot = pivot.sort_values("Logistic Regression", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Catch Rate (0=missed, 1=caught)"}
    )
    ax.set_title("Per-Attack-Type Catch Rate — EXP0 (Control Features)\nTrain: NSL-KDD → Test: CICIDS 2017",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Attack Type", fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig("results/figures/chart4_attack_heatmap_exp0.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart4_attack_heatmap_exp0.png")
    print("  Significance: Shows which specific attack types are caught vs missed.")
    print("  HTTP-based DoS attacks (Hulk, GoldenEye, Slowloris) expected near-zero catch rate.")
    print("  Maps directly to protocol-level explanations in the paper.")


# ══════════════════════════════════════════════════════
# CHART 5 — Attack Catch Rate EXP0 vs EXP2 (rst_ratio effect)
# Shows how adding rst_ratio changed per-attack detection
# KEY FINDING: rst_ratio helps PortScan but kills everything else
# ══════════════════════════════════════════════════════
def chart5_attack_exp0_vs_exp2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        exp0_data = breakdown[(breakdown["experiment"] == "exp0") & 
                               (breakdown["model"] == model)].set_index("attack_type")
        exp2_data = breakdown[(breakdown["experiment"] == "exp2") & 
                               (breakdown["model"] == model)].set_index("attack_type")

        common = exp0_data.index.intersection(exp2_data.index)
        exp0_vals = exp0_data.loc[common, "catch_rate"]
        exp2_vals = exp2_data.loc[common, "catch_rate"]
        diff = exp2_vals - exp0_vals

        colors = ["#4CAF50" if d >= 0 else "#F44336" for d in diff]
        bars = ax.barh(common, diff, color=colors, alpha=0.85)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("Change in Catch Rate\n(EXP2 − EXP0)")
        if idx == 0:
            ax.set_ylabel("Attack Type")

    fig.suptitle("Change in Per-Attack Catch Rate After Adding rst_ratio (EXP0 → EXP2)",
                 fontsize=13, fontweight="bold", y=1.02)

    green = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Improved")
    red   = mpatches.Patch(color="#F44336", alpha=0.85, label="Degraded")
    fig.legend(handles=[green, red], loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig("results/figures/chart5_attack_exp0_vs_exp2.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart5_attack_exp0_vs_exp2.png")
    print("  Significance: Shows rst_ratio's per-attack impact.")
    print("  Green bars = rst_ratio helped detect that attack type.")
    print("  Red bars = rst_ratio caused that attack type to be missed more.")


# ══════════════════════════════════════════════════════
# CHART 6 — Top Feature Importances EXP0 vs EXP2 (RF)
# Shows what RF relied on before and after rst_ratio was added
# KEY FINDING: rst_ratio becomes dominant in EXP2 → misleads model
# ══════════════════════════════════════════════════════
def chart6_rf_feature_importance():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, exp in enumerate(["exp0", "exp2"]):
        ax = axes[idx]
        df = importance[(importance["model"] == "rf") & 
                        (importance["experiment"] == exp)].copy()
        df = df.dropna(subset=["importance"])
        df = df.nlargest(8, "importance").sort_values("importance")

        bars = ax.barh(df["feature"], df["importance"], color="#4CAF50", alpha=0.8)
        ax.set_title(f"RF Feature Importance — {exp.upper()}", fontweight="bold")
        ax.set_xlabel("Importance Score")

        for bar in bars:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.3f}", va="center", fontsize=9)

    fig.suptitle("Random Forest — Top 8 Features: EXP0 (Control) vs EXP2 (+rst_ratio)",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig("results/figures/chart6_rf_importance_exp0_vs_exp2.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart6_rf_importance_exp0_vs_exp2.png")
    print("  Significance: Shows how adding rst_ratio shifts what RF relies on.")
    print("  If rst_ratio appears in top features in EXP2, it confirms the model")
    print("  committed to a feature that is near-zero for all attacks in CICIDS.")


# ══════════════════════════════════════════════════════
# CHART 7 — LR Coefficients EXP0 (feature directionality)
# Shows which features push toward attack vs normal
# KEY FINDING: positive coef = model learned this = attack signal
# ══════════════════════════════════════════════════════
def chart7_lr_coefficients():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, exp in enumerate(["exp0", "exp2"]):
        ax = axes[idx]
        df = importance[(importance["model"] == "lr") & 
                        (importance["experiment"] == exp)].copy()
        df = df.dropna(subset=["coefficient"])
        df["abs_coef"] = df["coefficient"].abs()
        df = df.nlargest(8, "abs_coef").sort_values("coefficient")

        colors = ["#F44336" if c > 0 else "#2196F3" for c in df["coefficient"]]
        bars = ax.barh(df["feature"], df["coefficient"], color=colors, alpha=0.85)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_title(f"LR Coefficients — {exp.upper()}", fontweight="bold")
        ax.set_xlabel("Coefficient\n(+ = toward attack, − = toward normal)")

    red  = mpatches.Patch(color="#F44336", alpha=0.85, label="Pushes toward ATTACK")
    blue = mpatches.Patch(color="#2196F3", alpha=0.85, label="Pushes toward NORMAL")
    fig.legend(handles=[red, blue], loc="upper right", fontsize=10)
    fig.suptitle("Logistic Regression Coefficients: EXP0 vs EXP2",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig("results/figures/chart7_lr_coefficients_exp0_vs_exp2.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart7_lr_coefficients_exp0_vs_exp2.png")
    print("  Significance: Shows which features LR learned as attack signals.")
    print("  Positive coefficient = model learned this feature → attack.")
    print("  In CICIDS if that feature is reversed, model classifies attacks as normal.")


# ══════════════════════════════════════════════════════
# CHART 8 — Feature Importance Heatmap Across All Experiments (RF)
# Shows how feature reliance shifts across the 6 experiments
# KEY FINDING: importance shifts from volume features to flag features
# ══════════════════════════════════════════════════════
def chart8_rf_importance_heatmap():
    rf_imp = importance[(importance["model"] == "rf")].copy()
    rf_imp = rf_imp.dropna(subset=["importance"])

    pivot = rf_imp.pivot_table(
        index="feature", columns="experiment", values="importance", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=EXPS)
    pivot = pivot.loc[pivot.max(axis=1).nlargest(10).index]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="Blues",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Feature Importance"}
    )
    ax.set_title("Random Forest Feature Importance Across All Experiments\n(Top 10 Features)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Feature")
    ax.set_xticklabels([e.upper() for e in EXPS], rotation=0)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig("results/figures/chart8_rf_importance_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart8_rf_importance_heatmap.png")
    print("  Significance: Shows how RF's feature reliance evolves as features are swapped.")
    print("  Volume features (bytes, duration) dominate EXP0.")
    print("  Protocol features take over in later experiments — RF commits to era-specific signals.")


# ══════════════════════════════════════════════════════
# CHART 9 — Recall Comparison (Within vs Cross)
# Recall is the most critical metric for IDS
# KEY FINDING: cross-dataset recall collapses = attacks missed
# ══════════════════════════════════════════════════════
def chart9_recall_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in MODELS:
        df = results[results["model"] == model].sort_values("experiment")
        ax.plot(
            [EXP_LABELS[e] for e in EXPS],
            [df[df["experiment"] == e]["cross_recall"].values[0] for e in EXPS],
            marker="s", linewidth=2.5, markersize=7,
            color=COLORS[model], label=MODEL_LABELS[model], linestyle="--"
        )

    ax.set_title("Cross-Dataset Recall Across Experiments\n(Recall = % of actual attacks caught)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Cross-Dataset Recall")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=10)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(5.1, 0.51, "50%\nthreshold", fontsize=8, color="gray", alpha=0.7)

    plt.tight_layout()
    plt.savefig("results/figures/chart9_cross_recall.png", bbox_inches="tight")
    plt.close()
    print("Saved: chart9_cross_recall.png")
    print("  Significance: Recall = how many actual attacks were caught.")
    print("  In IDS, low recall = missed attacks = security breach.")
    print("  Shows that protocol features cause the model to miss MORE attacks.")

def chart10_per_attack_across_experiments():
    attack_types = breakdown["attack_type"].unique()

    for attack in attack_types:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        for model in MODELS:
            df = breakdown[
                (breakdown["model"] == model) & 
                (breakdown["attack_type"] == attack)
            ].sort_values("experiment")
            
            if df.empty:
                continue
                
            catch_rates = [
                df[df["experiment"] == e]["catch_rate"].values[0] 
                if len(df[df["experiment"] == e]) > 0 else None
                for e in EXPS
            ]
            
            ax.plot(
                [EXP_LABELS[e] for e in EXPS],
                catch_rates,
                marker="o", linewidth=2.5, markersize=7,
                color=COLORS[model], label=MODEL_LABELS[model]
            )

        ax.set_title(f"Catch Rate Across Experiments — {attack}",
                        fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Experiment (Feature Added)")
        ax.set_ylabel("Catch Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.legend(fontsize=10)

        # clean filename
        safe_name = attack.replace(" ", "_").replace("/", "_").replace("\\", "_")
        path = f"results/figures/chart10_attack_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating all visualizations...\n")
    chart1_cross_f1_trend()
    chart2_within_vs_cross_gap()
    chart3_performance_drop()
    chart4_attack_heatmap()
    chart5_attack_exp0_vs_exp2()
    chart6_rf_feature_importance()
    chart7_lr_coefficients()
    chart8_rf_importance_heatmap()
    chart9_recall_comparison()
    chart10_per_attack_across_experiments()
    print("\nAll charts saved to results/figures/")
    print("\nChart summary for paper:")
    print("  chart1 → Main result: generalization degrades with protocol features")
    print("  chart2 → Benchmark illusion: within F1 misleads")
    print("  chart3 → rst_ratio caused most damage")
    print("  chart4 → Which attacks are caught/missed (heatmap)")
    print("  chart5 → rst_ratio's per-attack impact")
    print("  chart6 → What RF relied on before/after rst_ratio")
    print("  chart7 → LR feature directionality")
    print("  chart8 → How feature reliance shifts across experiments")
    print("  chart9 → Recall collapse — attacks being missed")