from sklearn.neighbors import LocalOutlierFactor


class LOF:
    def fit(self, X):
        self._lof = LocalOutlierFactor(n_neighbors=16, n_jobs=-1).fit(X[:4096])
        return self

    def anomaly_scores(self, X):
        return -self._lof._decision_function(X)
