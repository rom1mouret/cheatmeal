import pandas as pd
import numpy as np


class AnomalyEvaluator:
    def __init__(self, label_col, normal_classes, anomaly_classes, precision_k=None, chunksize=16384, strategy=np.mean):
        self._label_col = label_col
        self._normal_classes = normal_classes
        self._anomaly_classes = anomaly_classes
        self._precision_k = precision_k
        self._chunksize = chunksize
        self._strategy = strategy

    def evaluate(self, filename, trained_preproc, trained_detectors, skiprows, gpu_device=-1, delimiter=","):
        all_scores = []
        y_true = []

        relevant_classes = self._normal_classes + self._anomaly_classes
        for chunk in pd.read_csv(filename, chunksize=self._chunksize, skiprows=skiprows,
                                 header=None, delimiter=delimiter):
            relevant = chunk[chunk[self._label_col].isin(relevant_classes)]
            if len(relevant) == 0:
                continue

            # reference label
            labels = relevant[self._label_col].values
            binary = [0 if str(l) in self._normal_classes else 1 for l in labels]
            y_true += binary

            # scoring
            batch = relevant.values
            preprocessed = trained_preproc.transform(batch)
            ensemble = np.vstack([d.anomaly_scores(preprocessed, gpu_device) for d in trained_detectors])
            scores = self._strategy(ensemble, axis=0)
            all_scores.append(scores)

        all_scores = np.concatenate(all_scores)

        # evaluation
        y_true = np.array(y_true, dtype=np.int32)
        if self._precision_k is None:
            self._precision_k = np.sum(y_true)
        ranking = all_scores.argsort()[::-1][:self._precision_k]
        precision = np.sum(y_true[ranking])/self._precision_k

        self._precision = precision
        self._total = len(all_scores)

    def report(self):
        return "precision-at-%i/%i = %f%%" % (self._precision_k, self._total, 100*self._precision)
