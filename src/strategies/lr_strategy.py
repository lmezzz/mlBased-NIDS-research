import numpy as np
import pandas as pd

from src.strategies.base import MLStrategy


class LogisticRegressionStrategy(MLStrategy):
    name = "LogisticRegression"
    pkl_prefix = "lr"

    def decision(self, X: pd.DataFrame) -> np.ndarray:
        self.load()
        proba = self.model.predict_proba(self._scale(X))
        return proba[:, 1]
