from sklearn.preprocessing import StandardScaler
from sklearn.neighbors.kde import KernelDensity


class KDE:

    def __init__(self, scaling=True):
        self._scaling = scaling

    def fit(self, X):
        if self._scaling:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        X = X[:512]

        self._kde = KernelDensity().fit(X)

        return self

    def anomaly_scores(self, batch):
        if self._scaling:
            batch = self._scaler.transform(batch)

        return -self._kde.score_samples(batch)
