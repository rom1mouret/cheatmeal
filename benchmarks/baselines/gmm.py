from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


class GMM:

    def __init__(self, scaling=True, n_components=6):
        self._scaling = scaling
        self._n_components = n_components

    def fit(self, X):
        X = X[:20000]
        if self._scaling:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self._gmm = GaussianMixture(n_components=self._n_components).fit(X)

        return self

    def anomaly_scores(self, batch):
        if self._scaling:
            batch = self._scaler.transform(batch)

        return -self._gmm.score_samples(batch)
