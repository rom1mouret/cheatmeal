from sklearn.preprocessing import StandardScaler
from sklearn import svm


class OneClassSVM:

    def __init__(self, scaling=True):
        self._scaling = scaling

    def fit(self, X):
        if self._scaling:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        X = X[:4096]

        self._svm = svm.OneClassSVM().fit(X)

        return self

    def anomaly_scores(self, batch):
        if self._scaling:
            batch = self._scaler.transform(batch)

        return -self._svm.decision_function(batch)
