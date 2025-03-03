# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


class kFingerprinting:
    def __init__(
        self,
        random_state: int = 0,
        n_components: int = 10,
        n_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.scaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "kFingerprinting":
        self.model = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(X)), n_jobs=2
        )
        self.pca = PCA(n_components=min(self.n_components, len(X)))
        X = np.asarray(X)
        y = np.asarray(y)

        # scale
        X = self.scaler.fit_transform(X)

        # reduce
        Xred = self.pca.fit_transform(X)

        # classify
        self.model.fit(Xred, y)
        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)

        X = self.scaler.transform(X)
        Xred = self.pca.transform(X)

        return self.model.predict(Xred, *args, **kwargs)

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)
        X = self.scaler.transform(X)
        Xred = self.pca.transform(X)

        return self.model.predict_proba(Xred, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "k-fingerprinting"
