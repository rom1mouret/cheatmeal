import numpy as np


class Discretizer:
    """ use scikit's KBinsDiscretizer if ever released """
    def __init__(self, dense=True):
        assert dense, "only dense output is supported"

    def fit(self, X):
        self._bins = []
        for j in range(X.shape[1]):
            count, edges = np.histogram(X[:, j])
            self._bins.append(edges)

        return self

    def transform(self, X):
        new_cols = []
        for j, edges in enumerate(self._bins):
            discrete = np.zeros((X.shape[0], len(edges)+1))
            bin_index = np.digitize(X[:, j], edges)
            discrete[np.arange(X.shape[0]), bin_index] = 1
            new_cols.append(discrete)

        return np.concatenate(new_cols, axis=1)
