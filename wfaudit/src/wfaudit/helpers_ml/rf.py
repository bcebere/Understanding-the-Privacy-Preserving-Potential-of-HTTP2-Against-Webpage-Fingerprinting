# stdlib
from typing import Any, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# MACHINE LEARNING BENCHMARKS #####
class RFClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = 4,
        random_state: int = 0,
        n_estimators: int = 100,
        max_features="sqrt",
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        n_jobs: int = 4,
    ) -> None:
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict_proba(X)

    @staticmethod
    def name() -> str:
        return "random_forest"
