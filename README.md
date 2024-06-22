# Cheat Meal Autoencoders

Improvements over autoencoders in PyTorch.

CheatMeal library provides 3 classes:


* **Training Proxy** trains multiple autoencoders on multiple GPUs. It takes arbitrary-sized batches as input, delivered asynchronously by a process or a pool of processes spawned by the user.
* **CheatMealAE** allows its autoencoder to be forgiving on the reconstruction error of a small subset of columns. Anomaly detection takes advantage of this feature.
* **Recoder** places constraints on the latent units in order to facilitate classification via dimensionality reduction.

Both CheatMealAE and Recoder compare favorably to vanilla autoencoders for anomaly detection and dimensionality reduction respectively.

## Dimensionality Reduction Benchmarks

SVM classification performance averaged over 47 rounds from 3 rows randomly sampled from each class.
Dataset: [creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

| Algorithm       | Macro F1 | Macro F1 std  |
| ------------- |-------------:| -----:|
| Baseline Autoencoder     | 34.0% | 16.1%  |
| Recoder      |   76.4%     |   11.5% |

## Outlier Detection Benchmarks

Precision of a 3-detector ensemble averaged over 100 rounds.

Dataset: [kddcup99](http://kdd.ics.uci.edu/databases/kddcup99), DoS attacks are excluded because they outnumber normal data.

| Algorithm       | precision@k | precision@k std  | precision@k 10th percentile |
| ------------- |-------------:| -----:|-----:|
| CheatMeal (forgiveness=0) | 69.0% | 3.6%  | 65.5% |
| CheatMeal (forgiveness=2) |   73.4%     |   5.1% | 66.2% |
| Scikit IForest | 71.6% | 7.4% | 63.1 % |

`k ~ 8200 out of 200,000 rows`

CheatMealAE with `forgiveness=0` is equivalent to a regular autoencoder.
It should be noted that any improvement of the `forgiveness=0` baseline, such as loss customization, usually improves `forgiveness=2` AE just as much.

Compared to the recoder benchmarks, the margin is quite tight, so I ran a signed test.
The graphic reads as "the null hypothesis for a +3% margin has a low probability thus it can be rejected, therefore the improvement is real and over +3%."

<div align="center">
<img src="https://github.com/rom1mouret/cheatmeal/blob/master/kdd_signed_test.png">
</div>

## Recoder

When training a recoder, input data and backpropagation flow through the encoder twice. Hence the name.
The first pass over the data is the usual `input->latent->output` flow, to which corresponds the standard MSE/L1 reconstruction error, first term of the loss.
During the second pass, latent units are perturbed with Gaussian noise, decoded and re-encoded to latent values. The loss incorporates a second error term to account for the distance between the original latent representation and the noisy one.
The rationale is twofold:
1. Make the autoencoder robust to noise
2. Tie encoder and decoder together in that the decoder is forced to generate data that are somewhat recursively encodable, regardless of the reconstruction error in the original feature space.

Under the hood, we begin by encoding features to a 3D tensor of size `batch_size x latent_dim x 1`. The tensor is duplicated on its 3rd dimension and perturbed with zero-mean Gaussian noise of shape `batch_size x latent_dim x timesteps`.
The noisy latent units are then decoded, encoded again and finally compared to the noise-free latent representation in the loss function.
The number of timesteps is anything between 4 and 32. 3D tensors are processed using convolutions with `kernel_size=1`.

Note: only MSE and L1 losses are supported for now. CheatMealAE supports more loss functions.

<div align="center">
<img src="https://github.com/rom1mouret/cheatmeal/blob/master/recoder.png" width="400">
</div>

## CheatMealAE

CheatMealAE's difference with vanilla autoencoders lies in the decoding part: CheatMealAE has two decoders.
One decoder reconstructs the input, as in a standard autoencoder. Another decoder generates a tensor of the same size as the output. It determines how much each feature individually contributes to the loss and the anomaly score.

<div align="center">
<img src="https://github.com/rom1mouret/cheatmeal/blob/master/forgiver.png" width="370">
</div>

The rationale behind CheatMealAE is also twofold:
1. Reconstruct inputs more faithfully, for the part that is reconstructible. By assigning small weights to poorly-reconstructible features, the autoencoder eliminates a major hindrance and makes room for fitting other features more closely.
2. Lower the weights of the features that are random or random-*ish*, both in the loss and the anomaly score.

Another, more hypothetical reason can be given for the second rationale:
Granted latent units are soft assignments to clusters, mapping such latent units to large-weighed features LWF and to small-weighed features SWF implies that the corresponding cluster is located at LWF and spreads along the SWF axes. If we knew how SWF were spread, we could use CheatMealAE as a generative model.

CheatMealAE has an extra hyperparameter called 'forgiveness'. At `forgiveness=0`, CheatMealAE behaves like a normal autoencoder. At `forgiveness=1`, the loss can ignore up to 1 feature, i.e. `weights = 1 - softmax(decoded)`. At `forgiveness=2`, two softmax are blended together, and so on.


## Installation

Install dependencies:

* Python >= 3.4
* NumPy >= 1.11
* Scikit-learn >= 0.19
* PyTorch == 0.3.1 or 0.3.0
* Pandas >= 0.22 (for benchmarks only)

Now `cd` to cheatmeal directory and run `python3 setup.py install` with the adequate permissions.

## How To Reproduce The Results

For dimensionality reduction benchmarks, download the credit card fraud dataset from [Kaggle website](https://www.kaggle.com/mlg-ulb/creditcardfraud).

`cd` to benchmarks directory and run ./creditcard_dimred.bash creditcard.csv

For anomaly detection, download the full kddcup 99 dataset from [UCI website](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

`cd` to benchmarks directory and run ./kdd99_outlier_detect.bash kddcup.data

There is a bunch of other benchmark scripts for various datasets in the root directory. I hope script names are self-explanatory.
