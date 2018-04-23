#!/usr/bin/env python3

from cheatmeal.training_proxy import TrainingProxy
from cheatmeal.cheat_meal_ae import CheatMealAE

from preproc.preprocessor import Preprocessor
from .anomaly_evaluator import AnomalyEvaluator
from baselines.iforest import IForest

import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmarks for Outlier Detection')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs=None, help="shuffled CSV dataset")
    parser.add_argument('--split', metavar="index", type=int, nargs=None, help="where the training data ends and where it starts", required=True)
    parser.add_argument('--gpus', metavar="device-id", type=int, nargs='*', help="optional list of GPU device ids", default=[])
    parser.add_argument('--topk', metavar='k', type=int, nargs=None, help="the 'k' in precison@k (the evaluation metrics)")
    parser.add_argument('--ensemble', metavar='n', type=int, nargs=None, default=3, help="number of autoencoders to be ensembled")
    parser.add_argument('--forgiveness', metavar='f', type=int, nargs=None, default=-1, help="how many output units can be ignored in the loss function")
    parser.add_argument('--merge', metavar='stategy', type=str, nargs=None, default="harmonic", help="how to merge autoencoders in thje ensemble")
    parser.add_argument('--cat-cols', metavar='column', type=int, nargs='*', default=[], help="list of categorical column indices")
    parser.add_argument('--num-cols', metavar='column', type=int, nargs='*', default=[], help="list of numerical column indices, excluding binary columns")
    parser.add_argument('--binary-cols', metavar='column', type=int, nargs='*', default=[], help="list of binary column indices")
    parser.add_argument('--label-col', metavar='column', type=int, nargs=None, default=-1, help="index of the label column")
    parser.add_argument('--normal-classes', metavar='name', type=str, nargs='+', help="list of normal classes")
    parser.add_argument('--anomaly-classes', metavar='name', type=str, nargs='+', help="list of anomaly classes")
    parser.add_argument('--epochs', metavar="num", type=int, default=1, help="number of epochs per autoencoder")
    parser.add_argument('--loss', metavar="function", type=str, default="cross_entropy", help="loss function for training autoencoders")
    parser.add_argument('--delimiter', metavar="char", type=str, default=",", help="CSV delimiter")
    parser.add_argument('--has-header', action='store_true', help="whether the CSV file has a header")

    # options
    args = vars(parser.parse_args())
    set_filename = args['dataset']
    desired_cut = args['split']
    gpus = args['gpus']
    loss = args["loss"]
    precision_k = args.get("topk")
    normal_classes = args["normal_classes"]
    anomaly_classes = args["anomaly_classes"]
    cat_cols = args["cat_cols"]
    num_cols = args["num_cols"]
    bin_cols = args["binary_cols"]
    label_col = args["label_col"]
    forgiveness = args["forgiveness"]
    n_encoders = args["ensemble"]
    epochs = args["epochs"]
    delimiter = args["delimiter"]
    skiprows = 1 if args["has_header"] else 0

    if len(gpus) > 0 and len(gpus) != n_encoders:
        print("warning: it is recommended to specify as many GPUs as "
              "auto-encoders for them to go at the same speed. "
              "Using the same GPU for different autoencoders is allowed.")

    if forgiveness < 0:
        # arbitrary, not backed up by theory
        forgiveness = int(np.log(len(num_cols) + len(bin_cols) + 3*len(cat_cols)))
        print("setting forgiveness:", forgiveness)

    if label_col == -1:
        label_col = max(cat_cols + num_cols + bin_cols) + 1
    if args["merge"] in ("avg", "average", "mean"):
        merge = np.mean
    elif args["merge"] == "min":
        merge = np.min
    elif args["merge"] == "max":
        merge = np.max
    elif args["merge"] == "pseudoharmonic":
        def merge_func(arr, axis):
            epsed = arr + 0.000001
            return np.sum(np.log(epsed), axis=axis) - np.log(np.sum(epsed, axis=axis))
        merge = merge_func
    elif args["merge"] == "harmonic":
        def merge_func(arr, axis):
            epsed = arr + 0.000001
            return 1/np.sum(1/epsed, axis=axis)
        merge = merge_func
    else:
        print("unknown merging strategy", args["merge"])
        exit(1)

    chunksize = min(131072, desired_cut)
    cut = desired_cut - desired_cut % chunksize
    print("actual cut", cut)

    # train preprocessor
    preproc = Preprocessor(cat_cols, bin_cols, num_cols, normal_classes, label_col)
    preproc.train(set_filename, delimiter, skiprows=skiprows)
    num_cols = preproc.num_cols()

    # build auto-encoders
    encoders = []
    for i in range(n_encoders):
        if forgiveness > 0:
            f = i
        else:
            f = 0
        ae = CheatMealAE(non_binary_columns=num_cols, loss_type=loss, verbose=True, forgiveness=f)
        encoders.append(ae)

    # train auto-encoder
    learner = TrainingProxy(verbose=True)
    learner.start_training(encoders, gpu_devices=gpus)
    for epoch in range(epochs * len(encoders)):
        print("epoch", epoch)
        processed = 0
        for chunk in pd.read_csv(set_filename, chunksize=chunksize, header=None, delimiter=delimiter, skiprows=skiprows):
            processed += len(chunk)
            normal_data = chunk[chunk[label_col].isin(normal_classes)].values
            if len(normal_data) > 0:
                batch = preproc.transform(normal_data)
                batch = shuffle(batch)  # useful only if NN batches are smaller than chunksize
                learner.train_on_batch(batch)
                if processed+skiprows >= cut:
                    break

    # collect trained models
    print("stopping training")
    modelfiles = learner.stop_training(model_selection=False)
    print("training stopped")

    autoencoders = []
    for fname in modelfiles:
        ae = CheatMealAE(non_binary_columns=num_cols, loss_type=loss)
        ae.deserialize(fname)
        autoencoders.append(ae)

    # evaluate the ensemble
    evaluator = AnomalyEvaluator(label_col, normal_classes, anomaly_classes, precision_k, strategy=merge)
    evaluator.evaluate(set_filename, preproc, autoencoders, cut, delimiter=delimiter)
    print(evaluator.report(), "for forgiveness", forgiveness)

    # compare with IForest
    learner = IForest().fit(batch)
    evaluator.evaluate(set_filename, preproc, [learner], cut, delimiter=delimiter)
    print(evaluator.report(), "for IForest")
