# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler


class SVMClassifier:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str = "scale",
        probability: bool = True,
    ) -> None:
        self.model = svm.SVC(
            C=C, kernel=kernel, degree=degree, gamma=gamma, probability=probability
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SVMClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        self.scaler = MinMaxScaler().fit(X)
        X = self.scaler.transform(X)

        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)
        X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)
        X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    @staticmethod
    def name() -> str:
        return "support_vector_machine"
