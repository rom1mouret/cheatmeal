from sklearn.ensemble import IsolationForest
import numpy as np


class IForest:
    def __init__(self, max_samples="auto", n_jobs=-1):
        self._max_samples = max_samples
        self._n_jobs = n_jobs

    def fit(self, X):
        self._iforest = IsolationForest(max_samples=self._max_samples, n_jobs=self._n_jobs).fit(X)
        return self

    def anomaly_scores(self, batch, gpu_device):
        return -self._iforest.decision_function(batch)

    def increment_leaves_count(self, batch):
        for tree in self._iforest.estimators_:
            leaves_index = tree.apply(batch)
            count = np.bincount(leaves_index)
            loc = np.arange(np.max(leaves_index)+1)
            tree.tree_.n_node_samples[loc] += count

        self._iforest._max_samples += len(batch)
