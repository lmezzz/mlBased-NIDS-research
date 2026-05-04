from functools import lru_cache
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from config import RESULTS_DIR
from src.facade import DatasetFacade
from src.strategies import MLStrategy, get_strategy


class IDSContext:
    """
    Orchestrator that pairs a Strategy with the DatasetFacade.

    This is the single entry point used by the Flask layer. It hides the
    fact that prediction reads a saved .pkl while metrics read a CSV.
    """

    def __init__(self, strategy: MLStrategy, facade: DatasetFacade | None = None):
        self.strategy = strategy
        self.facade = facade or DatasetFacade()

    @classmethod
    def for_(cls, model: str, experiment: str) -> "IDSContext":
        return cls(get_strategy(model, experiment))

    # ── Live prediction ──
    def predict_row(self, features: dict) -> dict:
        df = pd.DataFrame([features])
        preds, scores = self.strategy.predict_with_confidence(df)
        return {
            "prediction": int(preds[0]),
            "label": "ATTACK" if int(preds[0]) == 1 else "BENIGN",
            "score": float(scores[0]),
            "model": self.strategy.name,
            "experiment": self.strategy.experiment,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        preds, scores = self.strategy.predict_with_confidence(df)
        out = df.copy()
        out["prediction"] = preds
        out["score"] = scores
        out["label"] = ["ATTACK" if p == 1 else "BENIGN" for p in preds]
        return out

    # ── Evaluation against a prepared dataset ──
    def evaluate(self, dataset: str) -> dict:
        X, y = self.facade.load_and_prepare(dataset, self.strategy.experiment)
        preds, _ = self.strategy.predict_with_confidence(X)
        return {
            "dataset": dataset,
            "n_rows": int(len(y)),
            "f1": float(f1_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
        }

    def feature_schema(self) -> list[dict]:
        self.strategy.load()
        return [
            {"name": col, "numeric": col in self.strategy.numeric_cols}
            for col in self.strategy.feature_cols
        ]


# ── Cached, pkl-free read of the published results table ──
@lru_cache(maxsize=1)
def load_results_table() -> pd.DataFrame:
    """Read the all-experiments metrics CSV. Used for the browse-results view."""
    candidates = [RESULTS_DIR / "experiment_results.csv", RESULTS_DIR / "all_results.csv"]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No results CSV found in {RESULTS_DIR}")


def lookup_results(model: str, experiment: str) -> dict | None:
    df = load_results_table()
    model_map = {"lr": "LogisticRegression", "rf": "RandomForest", "svm": "SVM"}
    full_name = model_map.get(model.lower(), model)
    row = df[(df["model"] == full_name) & (df["experiment"] == experiment.upper())]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()}


def attack_breakdown(model: str, experiment: str) -> pd.DataFrame:
    p = RESULTS_DIR / "attack_breakdown_all.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[(df["model"] == model.lower()) & (df["experiment"] == experiment.lower())]


def feature_importance(model: str, experiment: str) -> pd.DataFrame:
    p = RESULTS_DIR / "feature_importance" / f"{model.lower()}_importance_{experiment.lower()}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def list_figure_files() -> list[str]:
    figures_dir = RESULTS_DIR / "figures"
    if not figures_dir.exists():
        return []
    return sorted(p.name for p in figures_dir.glob("*.png"))
