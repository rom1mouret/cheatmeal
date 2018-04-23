from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


class PCADetector:

    def __init__(self, scaling=True, max_components=6, ratio=20):
        self._scaling = scaling
        self._ratio = ratio
        self._max_components = max_components

    def fit(self, X):
        if self._scaling:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        output_dim = max(2, min(self._max_components, X.shape[1]//self._ratio))
        self._pca = PCA(n_components=output_dim).fit(X)

        return self

    def anomaly_scores(self, batch):
        if self._scaling:
            batch = self._scaler.transform(batch)
        X = self._pca.transform(batch)
        approx = self._pca.inverse_transform(X)

        return np.mean((batch-approx)**2, axis=1)
