import multiprocessing as mp
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import tempfile


mp = mp.get_context("spawn")


class Net(nn.Module):
    def __init__(self, input_dim, latent_dim, forgiveness, gpu_device):
        super(Net, self).__init__()

        hidden_dim = input_dim + 2
        self._input_dim = input_dim
        self._forgiveness = forgiveness

        # 'foriviving' weights when forgiveness = 0
        self.register_buffer("_default_weights", torch.ones(1, input_dim))

        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True),
            nn.Linear(hidden_dim, latent_dim)
        )

        self._decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim, affine=True),
            nn.Linear(latent_dim, input_dim)
        )

        if self._forgiveness >= 1:
            self._forgiver = nn.Sequential(
                nn.BatchNorm1d(latent_dim, affine=True),
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, affine=True),
                nn.Linear(hidden_dim, forgiveness * input_dim)
            )

    def encode(self, X):
        return self._encoder(X)

    def decode(self, latent):
        return self._decoder(latent)

    def forgive_weights(self, latent):
        if self._forgiveness == 0:
            return Variable(self._default_weights)

        raw_w = self._forgiver(latent)
        three_dim = raw_w.view(raw_w.size(0), self._input_dim, -1)
        normalized = nn.functional.softmax(three_dim, dim=1)
        combined = (1 - normalized).prod(dim=2)
        combined = combined / combined.sum(dim=1, keepdim=True)

        return combined


def binarize_targets(y_pred, y_true, not_bin_cols, bin_cols):
    if len(not_bin_cols) > 0:
        not_binary = y_true[:, not_bin_cols]
        binarized = nn.functional.sigmoid(0.1*not_binary)

        if len(bin_cols) > 0:
            binary = y_true[:, bin_cols]
            y_true = torch.cat([binarized, binary], dim=1)
        else:
            y_true = binarized

    # same order for the predictions
    # (doesn't really matter if y_pred is not used outside this function)
    if y_pred is not None:
        reordering = np.concatenate([not_bin_cols, bin_cols])
        y_pred = y_pred[:, reordering]

    return y_pred, y_true


def cross_entropy(y_pred, y_true, not_bin_cols, bin_cols):
    y_pred, y_true = binarize_targets(y_pred, y_true, not_bin_cols, bin_cols)

    max_val = (-y_pred).clamp(min=0)
    E = y_pred - y_pred * y_true + max_val + ((-max_val).exp() + (-y_pred - max_val).exp()).log()

    return E


class CheatLoss(nn.Module):

    def __init__(self, loss_type, not_binary_cols, binary_cols):
        super(CheatLoss, self).__init__()
        self._loss_type = loss_type.lower()
        self._not_bin_cols = not_binary_cols
        self._bin_cols = binary_cols

    def forward(self, y_pred, y_true, forgive_weights):
        if self._loss_type == "mse":
            diff = (y_true - y_pred)**2
        elif self._loss_type == "abs":
            diff = (y_true - y_pred).abs()
        elif self._loss_type == "huber":
            diff = nn.SmoothL1Loss(reduce=False)(y_pred, y_true)
        elif self._loss_type == "cross_entropy":
            diff = cross_entropy(y_pred, y_true, self._not_bin_cols, self._bin_cols)
        elif self._loss_type == "cosine":
            y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)  # should it be weighted?
            y_true = y_true / y_true.norm(dim=1, keepdim=True)
            diff = 1 - y_pred * y_true

        loss = (diff * forgive_weights).mean()

        # compute scores that we'll use to compare models during model selection
        # gradient information is dropped on purpose
        ungrad_diff = Variable(diff.data)
        ungrad_weights = Variable(forgive_weights.data)

        weights = ungrad_weights.sqrt()
        weights = weights / weights.sum(dim=1, keepdim=True)
        scores = (ungrad_diff * weights).mean()

        return loss, scores


class CheatMealAE:
    def __init__(self, loss_type="cross_entropy", non_binary_columns=[], verbose=False, forgiveness=None, latent_dim=None, lr=0.05):
        self._verbose = verbose
        self._not_bin_cols = non_binary_columns
        self._latent_dim = latent_dim
        self._lr = lr
        self._forgiveness = forgiveness

        if loss_type.lower() not in ("mse", "abs", "huber", "cross_entropy", "cosine"):
            print("warning: unknown loss type '%s'. Using cross_entropy instead" % loss_type)
            loss_type = "cross_entropy"

        self._loss_type = loss_type

    def start_training(self, feeding_q, gpu_device):
        """ gpu_device = -1 for CPU """
        if self._verbose:
            print("start AE on device:", gpu_device)

        self._exit_msg = self.__hash__()  # not super-safe but safe enough

        self._q = feeding_q
        self._serialized_net_file = mp.SimpleQueue()
        self._eval_score = mp.SimpleQueue()
        self._train_process = mp.Process(target=self._train, args=(gpu_device, ))
        self._train_process.start()

    def prepare_batch(self, batch):
        return torch.from_numpy(batch.astype(np.float32))

    def _train(self, gpu_device):
        net = None
        cudnn.benchmark = True
        np_loss = None

        while True:
            b = self._q.get()  # stop untils not-empty
            if type(b) == int:
                if b == self._exit_msg:
                    if self._verbose:
                        print("received STOP request. Exiting subprocess")
                    break
                else:
                    self._q.put(b)
                    continue
            try:
                if net is None:
                    bin_cols = list(set(range(b.size(1))).difference(self._not_bin_cols))

                    if self._latent_dim is None:
                        # rather arbitrary, not backed up by theory
                        self._latent_dim = int(np.log(1 + b.size(1)))
                    if self._forgiveness is None:
                        self._forgiveness = int(np.log(b.size(1))/2)

                    if self._verbose:
                        print("[initializing NN] latent: %i, forgiveness: %.2f, lr: %f, bin_cols: %s, others: %s, loss: %s" %
                              (self._latent_dim, self._forgiveness, self._lr, bin_cols, self._not_bin_cols, self._loss_type))

                    net = Net(b.size(1), latent_dim=self._latent_dim, forgiveness=self._forgiveness, gpu_device=gpu_device)
                    if gpu_device >= 0:
                        net = net.cuda(gpu_device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=self._lr)
                    loss_func = CheatLoss(self._loss_type, self._not_bin_cols, bin_cols)

                if gpu_device >= 0:
                    b = b.cuda(gpu_device)

                b = Variable(b)

                # forward
                optimizer.zero_grad()
                latent = net.encode(b)
                y_pred = net.decode(latent)
                weights = net.forgive_weights(latent)

                # back-propagation
                loss, scores = loss_func(y_pred, b, weights)
                loss.backward()

                optimizer.step()
                val = float(scores.data.cpu().numpy()[0])

                if np_loss is None:
                    np_loss = val
                else:
                    np_loss = 0.8 * np_loss + 0.2 * val  # moving average

                if self._verbose:
                    val = float(loss.data.cpu().numpy()[0])
                    print("loss", val)

            except Exception as e:
                print("exception thrown while trying to backprop: %s. Ignoring it." % e)
                traceback.print_tb(e.__traceback__)
                time.sleep(0.5)
                continue

        # serialize model in temporary file
        model = net.cpu().eval()
        f = tempfile.NamedTemporaryFile(prefix="model_", delete=False)
        filename = f.name
        torch.save(model, filename)

        # send it to upper-level process
        self._serialized_net_file.put(filename)

        # send the eval score to upper-level process as well
        self._eval_score.put(np_loss)

    def deserialize(self, filename):
        self._net = torch.load(filename)

    def stop_training(self):
        if self._verbose:
            print("putting EXIT MSG in queue")
        self._q.put(self._exit_msg)
        if self._verbose:
            print("joining training process")
        self._train_process.join()
        if self._verbose:
            print("trained process joined")

        return self._serialized_net_file.get(), self._eval_score.get()

    def _to_cuda_var(self, tensor, gpu_device):
        v = Variable(tensor, volatile=True)
        if gpu_device >= 0:
            v = v.cuda(gpu_device)
        return v

    def anomaly_scores(self, X, gpu_device):
        X = Variable(self.prepare_batch(X), volatile=True)
        if gpu_device >= 0:
            net = self._net.cuda(gpu_device)
            X = X.cuda(gpu_device)
        else:
            net = self._net

        # reconstruct
        latent = net.encode(X)
        y_pred = net.decode(latent)
        weights = net.forgive_weights(latent)
        weights = weights.sqrt()
        weights = weights/weights.sum(dim=1, keepdim=True)

        if self._loss_type == "cosine":
            y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
            y_true = X / X.norm(dim=1, keepdim=True)
            diff = 1 - y_pred * y_true
        elif self._loss_type == "cross_entropy":
            bin_cols = list(set(range(X.size(1))).difference(self._not_bin_cols))
            diff = cross_entropy(y_pred, X, self._not_bin_cols, bin_cols)
        else:
            diff = (X - y_pred).abs()

        weighted_diff = (diff * weights).sum(dim=1)

        return weighted_diff.data.cpu().numpy()

    def reduce_dim(self, X, gpu_device):
        X = Variable(self.prepare_batch(X), volatile=True)
        if gpu_device >= 0:
            net = self._net.cuda(gpu_device)
            X = X.cuda(gpu_device)
        else:
            net = self._net

        latent = net.encode(X)

        return latent.data.cpu().numpy()

    def train_on_batch(self, batch):
        """ this method can be called concurrently from different processess """
        self._q.put(batch)
