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


def to_cuda_var(tensor, gpu_device, volatile=False):
    v = Variable(tensor, volatile=volatile)
    if gpu_device >= 0:
        v = v.cuda(gpu_device)
    return v


class Net(nn.Module):
    def __init__(self, input_dim, latent_dim, gpu_device):
        super(Net, self).__init__()

        hidden_dim = input_dim + 2
        self._input_dim = input_dim

        self._encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        )

        self._decoder = nn.Sequential(
            #nn.Tanh(),
            nn.BatchNorm1d(latent_dim, affine=True),
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        )

    def encode(self, X):
        return self._encoder(X)

    def expand_latent(self, latent, gpu_device, random_repeats=16):
        r = torch.randn(latent.size(0), latent.size(1), random_repeats)
        return latent + to_cuda_var(r, gpu_device)

    def decode(self, latent):
        return self._decoder(latent)


class CyclicLoss(nn.Module):

    def __init__(self, loss_type):
        super(CyclicLoss, self).__init__()
        self._loss_type = loss_type.lower()

    def _compute_diff(self, y_pred, y_true):
        if self._loss_type == "mse":
            diff = (y_true - y_pred)**2
        elif self._loss_type == "abs":
            diff = (y_true - y_pred).abs()

        return diff

    def forward(self, y_pred, y_true, original_latent, cycled_latent):
        # The loss has two components

        # 1. usual reconstruction error
        loss = self._compute_diff(y_pred, y_true).mean()

        # 2. the consistency of the latent units
        if cycled_latent is not None:
            #original_latent = Variable(original_latent.data) # ignore gradients
            loss = loss + 2.5 * self._compute_diff(cycled_latent, original_latent).mean()

        return loss


class Recoder:
    def __init__(self, loss_type="huber", verbose=False, control=False, latent_dim=None, lr=0.05):
        self._verbose = verbose
        self._latent_dim = latent_dim
        self._lr = lr
        self._control = control

        if loss_type.lower() not in ("mse", "abs"):
            print("warning: unknown loss type '%s'. Using ABS instead" % loss_type)
            loss_type = "abs"

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
                    if self._latent_dim is None:
                        # rather arbitrary, not backed up by theory
                        self._latent_dim = int(np.log(1 + 8*b.size(1)))

                    if self._verbose:
                        print("[initializing NN] latent: %i, lr: %f, loss: %s" % (self._latent_dim,  self._lr, self._loss_type))

                    net = Net(b.size(1), latent_dim=self._latent_dim, gpu_device=gpu_device)
                    if gpu_device >= 0:
                        net = net.cuda(gpu_device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=self._lr)
                    loss_func = CyclicLoss(self._loss_type)

                if gpu_device >= 0:
                    b = b.cuda(gpu_device)

                b = Variable(b).unsqueeze(2)

                # the usual forward
                optimizer.zero_grad()
                latent = net.encode(b)
                y_pred = net.decode(latent)

                # the unusual 'cycle'
                if self._control:
                    new_latent = None
                else:
                    random = net.expand_latent(latent, gpu_device)
                    reconstruct = net.decode(random)
                    new_latent = net.encode(reconstruct)

                # back-propagation
                loss = loss_func(y_pred, b, latent, new_latent)
                loss.backward()
                optimizer.step()

                val = float(loss.data.cpu().numpy()[0])
                if np_loss is None:
                    np_loss = val
                else:
                    np_loss = 0.8 * np_loss + 0.2 * val  # moving average

                if self._verbose:
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
            print("putting None in queue")
        self._q.put(self._exit_msg)  # might get consumed by another AE first, but order doesn't matter

        if self._verbose:
            print("joining process")

        self._train_process.join()

        if self._verbose:
            print("process joined")

        return self._serialized_net_file.get(), self._eval_score.get()

    def anomaly_scores(self, X, gpu_device):
        raise NotImplementedError()

    def reduce_dim(self, X, gpu_device):
        X = Variable(self.prepare_batch(X), volatile=True)
        if gpu_device >= 0:
            net = self._net.cuda(gpu_device)
            X = X.cuda(gpu_device)
        else:
            net = self._net

        X = X.unsqueeze(2)
        latent = net.encode(X).squeeze(2)

        return latent.data.cpu().numpy()

    def train_on_batch(self, batch):
        """ this method can be called concurrently from different processess """
        self._q.put(batch)
