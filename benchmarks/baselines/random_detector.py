import numpy as np

class RandomDetector:

    def fit(self, X):
        pass

    def anomaly_scores(self, batch):
        return np.random.uniform(batch.shape[0])
