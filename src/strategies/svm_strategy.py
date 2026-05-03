import numpy as np
import pandas as pd

from src.strategies.base import MLStrategy


class SVMStrategy(MLStrategy):
    name = "SVM"
    pkl_prefix = "svm"

    def decision(self, X: pd.DataFrame) -> np.ndarray:
        self.load()
        # LinearSVC has no predict_proba; decision_function returns signed distance.
        return self.model.decision_function(self._scale(X))
