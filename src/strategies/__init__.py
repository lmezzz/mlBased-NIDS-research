from src.strategies.base import MLStrategy
from src.strategies.lr_strategy import LogisticRegressionStrategy
from src.strategies.rf_strategy import RandomForestStrategy
from src.strategies.svm_strategy import SVMStrategy

REGISTRY = {
    "lr": LogisticRegressionStrategy,
    "rf": RandomForestStrategy,
    "svm": SVMStrategy,
}


def get_strategy(name: str, experiment: str) -> MLStrategy:
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[key](experiment)
