
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .categorical_encoder import CategoricalEncoder  # replace with scikit's version if scikit >= 0.20


class Preprocessor:
    def __init__(self, categories, binary_cols, num_cols, normal_classes, class_index, training_cells=262144):
        self._cat_cols = categories
        self._bin_cols = binary_cols
        self._num_cols = num_cols
        self._normal_classes = normal_classes
        self._label_col = class_index
        self._training_cells = training_cells

    def train(self, filename, delimiter=",", skiprows=0):
        # train one-hot on a subsample of the full dataset
        if len(self._cat_cols) > 0:
            training_rows = self._training_cells // (1 + len(self._cat_cols))
            for chunk in pd.read_csv(filename, chunksize=training_rows,
                                     usecols=self._cat_cols+[self._label_col],
                                     dtype=str, header=None, delimiter=delimiter,
                                     skiprows=skiprows):
                data = chunk[chunk[self._label_col].isin(self._normal_classes)][self._cat_cols].values
                self._one_hot = CategoricalEncoder(dense=True).fit(data)
                break

        # train scaler on the full dataset
        if len(self._num_cols) > 0:
            self._scaler = StandardScaler()
            training_rows = self._training_cells // (1 + len(self._num_cols))
            for chunk in pd.read_csv(filename, chunksize=training_rows, delimiter=delimiter,
                                     usecols=self._num_cols+[self._label_col], header=None,
                                     skiprows=skiprows):
                data = chunk[chunk[self._label_col].isin(self._normal_classes)][self._num_cols].values
                if len(data) > 0:
                    self._scaler.partial_fit(data)

        return self

    def transform(self, batch):
        empty = np.empty((batch.shape[0], 0))
        if len(self._cat_cols) > 0:
            X_cat = self._one_hot.transform(batch[:, self._cat_cols])
        else:
            X_cat = empty

        if len(self._num_cols) > 0:
            X_num = self._scaler.transform(batch[:, self._num_cols])
        else:
            X_num = empty

        X_bin = batch[:, self._bin_cols]

        return np.concatenate([X_num, X_bin, X_cat], axis=1)

    def cat_cols(self):
        raise NotImplementedError()

    def num_cols(self):
        return np.arange(len(self._num_cols))

    def bin_cols(self):
        return np.arange(len(self._bin_cols))+len(self._num_cols)
