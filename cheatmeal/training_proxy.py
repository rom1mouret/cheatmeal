import multiprocessing as mp
import numpy as np

mp = mp.get_context("spawn")


class TrainingProxy:
    def __init__(self, verbose=False):
        self._verbose = verbose

    def start_training(self, autoencoders, training_batch_size=4096, gpu_devices=[]):
        """ call this method from a top-level process """
        self._autoencoders = autoencoders
        self._training_batch_size = training_batch_size
        self._current_batch = []  # that will be copied in every process and not shared between them
        self._current_batch_size = 0  # ditto
        self._first_batches = []
        self._feeding_q = mp.Queue(len(autoencoders))
        for i, ae in enumerate(autoencoders):
            ae.start_training(self._feeding_q, gpu_devices[i] if i < len(gpu_devices) else -1)

    def stop_training(self, model_selection=False):
        """ call this method from same process as start_training """
        collected = [ae.stop_training() for ae in self._autoencoders]
        if self._verbose:
            for c in collected:
                print("score[%s] = %f" % c)

        fnames, scores = zip(*collected)

        if model_selection:
            best = np.argmin(scores)
            return [fnames[best]]

        return fnames

    def train_on_batch(self, batch):
        """ this method can be called concurrently from different processess """
        new_size = self._current_batch_size + batch.shape[0]
        if new_size >= self._training_batch_size:
            cut = batch.shape[0] - (new_size - self._training_batch_size)
            batch_1 = batch[:cut]
            batch_2 = batch[cut:]
            self._current_batch.append(batch_1)
            X = np.concatenate(self._current_batch, axis=0)
            self._current_batch = []
            self._current_batch_size = 0
            prepared = self._autoencoders[0].prepare_batch(X)  # run on CPU
            self._feeding_q.put(prepared)

            # recurse on the residual
            self.train_on_batch(batch_2)
        else:
            self._current_batch.append(batch)
            self._current_batch_size = new_size
