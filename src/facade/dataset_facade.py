from pathlib import Path
import pandas as pd

from config import (
    KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL,
    KDD_TRAIN_EXP1, KDD_TEST_EXP1, CICIDS_EXP1,
    KDD_TRAIN_EXP2, KDD_TEST_EXP2, CICIDS_EXP2,
    KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3,
    KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4,
    KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5,
)


# (kdd_train, kdd_test, cicids) per experiment
EXPERIMENT_PATHS: dict[str, tuple[Path, Path, Path]] = {
    "EXP0": (KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL),
    "EXP1": (KDD_TRAIN_EXP1, KDD_TEST_EXP1, CICIDS_EXP1),
    "EXP2": (KDD_TRAIN_EXP2, KDD_TEST_EXP2, CICIDS_EXP2),
    "EXP3": (KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3),
    "EXP4": (KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4),
    "EXP5": (KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5),
}

DATASET_TO_INDEX = {
    "kdd_train": 0,
    "kdd_test": 1,
    "cicids": 2,
}

EXPERIMENT_DESCRIPTIONS = {
    "EXP0": "Control (baseline) — protocol-unaware features only",
    "EXP1": "Swap src_bytes -> syn_ratio (SYN flood signal)",
    "EXP2": "Swap dst_bytes -> rst_ratio (port-scan signal)",
    "EXP3": "Add fin_ratio (connection lifecycle)",
    "EXP4": "Add data_pkt_ratio (payload presence)",
    "EXP5": "Add service_bucket one-hot (IANA tier)",
}


class DatasetFacade:
    """
    Facade over loader + cleaner + extractor + aligner + preprocessor.

    Callers ask for a (dataset, experiment) pair; the facade returns a
    ready-to-feed DataFrame that matches what the saved Strategy expects.

    Read-only: this class assumes the experiment CSVs already exist on disk
    (produced by run_*_pipeline functions in src/pipeline.py). It does not
    re-run the pipelines.
    """

    @staticmethod
    def list_experiments() -> list[str]:
        return list(EXPERIMENT_PATHS.keys())

    @staticmethod
    def list_datasets() -> list[str]:
        return list(DATASET_TO_INDEX.keys())

    @staticmethod
    def describe(experiment: str) -> str:
        return EXPERIMENT_DESCRIPTIONS.get(experiment.upper(), "")

    @classmethod
    def is_available(cls, dataset: str, experiment: str) -> bool:
        try:
            return cls._path(dataset, experiment).exists()
        except (KeyError, ValueError):
            return False

    @classmethod
    def _path(cls, dataset: str, experiment: str) -> Path:
        exp = experiment.upper()
        if exp not in EXPERIMENT_PATHS:
            raise ValueError(f"Unknown experiment '{experiment}'")
        ds = dataset.lower()
        if ds not in DATASET_TO_INDEX:
            raise ValueError(f"Unknown dataset '{dataset}'")
        return EXPERIMENT_PATHS[exp][DATASET_TO_INDEX[ds]]

    @classmethod
    def load_and_prepare(cls, dataset: str, experiment: str) -> tuple[pd.DataFrame, pd.Series]:
        """
        Single entry point for prepared data.

        Returns:
            (X, y) where X is the feature DataFrame and y the binary label Series.
        """
        path = cls._path(dataset, experiment)
        if not path.exists():
            raise FileNotFoundError(
                f"Prepared dataset not found: {path}. "
                f"Run the pipeline functions in src.pipeline first."
            )
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise ValueError(f"'label' column missing from {path}")
        y = df["label"].astype(int)
        X = df.drop(columns=["label"])
        return X, y

    @classmethod
    def sample_row(cls, dataset: str, experiment: str, kind: str = "attack") -> dict:
        """Pick one example row to pre-fill the prediction form."""
        X, y = cls.load_and_prepare(dataset, experiment)
        if kind == "benign":
            mask = y == 0
        else:
            mask = y == 1
        if not mask.any():
            return X.iloc[0].to_dict()
        return X.loc[mask].iloc[0].to_dict()
