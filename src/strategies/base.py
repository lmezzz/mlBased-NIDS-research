from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR


class MLStrategy(ABC):
    """
    Strategy interface for ML algorithms.

    Concrete strategies wrap a trained .pkl artifact saved by the legacy
    run_<model>_EXP<n> training scripts. Each artifact contains:
        {model, scaler, features, numeric_cols}

    The Strategy is read-only over the saved artifact and exposes a uniform
    predict / decision API to the IDSContext and Flask layer.
    """

    name: str = "MLStrategy"
    pkl_prefix: str = ""

    def __init__(self, experiment: str):
        self.experiment = experiment.upper()
        self._loaded = False
        self.model = None
        self.scaler = None
        self.feature_cols: list[str] = []
        self.numeric_cols: list[str] = []

    @property
    def pkl_path(self) -> Path:
        return MODELS_DIR / f"{self.pkl_prefix}_{self.experiment.lower()}.pkl"

    def is_available(self) -> bool:
        return self.pkl_path.exists()

    def load(self) -> "MLStrategy":
        if self._loaded:
            return self
        if not self.is_available():
            raise FileNotFoundError(
                f"Model artifact not found: {self.pkl_path}. "
                f"Run the legacy training script for {self.name} {self.experiment} "
                f"to generate it."
            )
        payload = joblib.load(self.pkl_path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_cols = list(payload["features"])
        self.numeric_cols = list(payload["numeric_cols"])
        self._loaded = True
        return self

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        if self.numeric_cols:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df[self.feature_cols]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.load()
        return self.model.predict(self._scale(X))

    @abstractmethod
    def decision(self, X: pd.DataFrame) -> np.ndarray:
        """Return a confidence score per row (probability or decision distance)."""

    def predict_with_confidence(self, X: pd.DataFrame):
        preds = self.predict(X)
        scores = self.decision(X)
        return preds, scores
