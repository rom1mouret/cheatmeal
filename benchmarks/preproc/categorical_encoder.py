from sklearn.preprocessing import OneHotEncoder
import numpy as np


class CategoricalEncoder:
    """ if scikit >= 0.20, better use scikit's version instead of this class """
    def __init__(self, dense=True):
        assert dense, "only dense output is supported"

    def fit(self, X):
        self._str_to_int = {}
        X_int = np.empty(X.shape, dtype=np.int32)

        for i, row in enumerate(X):
            for j, v in enumerate(row):
                int_v = self._str_to_int.get(v)
                if int_v is None:
                    int_v = len(self._str_to_int) + 1
                    self._str_to_int[v] = int_v
                X_int[i, j] = int_v

        self._one_hot = OneHotEncoder(sparse=False).fit(X_int)

        return self

    def transform(self, X):
        X_int = np.empty(X.shape, dtype=np.int32)
        for i, row in enumerate(X):
            for j, v in enumerate(row):
                X_int[i, j] = self._str_to_int.get(v, 0)

        return self._one_hot.transform(X_int)
