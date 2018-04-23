#!/usr/bin/env python3

from cheatmeal.training_proxy import TrainingProxy
from cheatmeal.recoder import Recoder

from preproc.preprocessor import Preprocessor

import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmarks for Dimensionality Reduction')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs=None, help="shuffled CSV dataset")
    parser.add_argument('--split', metavar="index", type=int, nargs=None, help="where the training data ends and where it starts", required=True)
    parser.add_argument('--gpu', metavar="device-id", type=int, nargs=None, help="optional list of GPU device id", default=-1)
    parser.add_argument('--cat-cols', metavar='column', type=int, nargs='*', default=[], help="list of categorical column indices")
    parser.add_argument('--num-cols', metavar='column', type=int, nargs='*', default=[], help="list of numerical columns indices")
    parser.add_argument('--binary-cols', metavar='column', type=int, nargs='*', default=[], help="list of binary columns indices")
    parser.add_argument('--label-col', metavar='column', type=int, nargs=None, default=-1, help="index of the label column")
    parser.add_argument('--classes', metavar='name', type=str, nargs='+', help="rows tagged with other classes will be ignored")
    parser.add_argument('--sampling', metavar='name', type=int, nargs=None, default=3, help="number of samples for each class used to train the classifier")
    parser.add_argument('--epochs', metavar="num", type=int, default=1, help="number of epochs per autoencoder")
    parser.add_argument('--loss', metavar="function", type=str, default="cross_entropy", help="loss function to train the autoencoder")
    parser.add_argument('--delimiter', metavar="char", type=str, default=",", help="CSV delimiter")
    parser.add_argument('--has-header', action='store_true', help="whether the CSV has a header")
    parser.add_argument('--control', metavar='0-or-1', type=int, nargs=None, default=0, help="if '1' then the script will benchmark a regular autoencoder")

    # options
    args = vars(parser.parse_args())
    set_filename = args['dataset']
    desired_cut = args['split']
    gpu = args['gpu']
    loss = args["loss"]
    sampling = args["sampling"]
    classes = args["classes"]
    cat_cols = args["cat_cols"]
    num_cols = args["num_cols"]
    bin_cols = args["binary_cols"]
    label_col = args["label_col"]
    epochs = args["epochs"]
    delimiter = args["delimiter"]
    control = args["control"] == 1
    skiprows = 1 if args["has_header"] else 0

    if label_col == -1:
        label_col = max(cat_cols + num_cols + bin_cols) + 1

    chunksize = min(131072, desired_cut)
    cut = desired_cut - desired_cut % chunksize
    print("actual cut", cut)

    # train preprocessor
    preproc = Preprocessor(cat_cols, bin_cols, num_cols, classes, label_col)
    preproc.train(set_filename, delimiter, skiprows=skiprows)
    num_cols = preproc.num_cols()

    # build auto-encoder
    ae = Recoder(loss_type=loss, control=control, verbose=True)

    # train auto-encoder
    learner = TrainingProxy(verbose=True)
    learner.start_training([ae], gpu_devices=[gpu])
    for epoch in range(epochs):
        print("epoch", epoch)
        processed = 0
        for chunk in pd.read_csv(set_filename, chunksize=chunksize, header=None, delimiter=delimiter, skiprows=skiprows):
            processed += len(chunk)
            relevant_data = chunk[chunk[label_col].isin(classes)].values
            if len(relevant_data) > 0:
                batch = preproc.transform(relevant_data)
                batch = shuffle(batch)  # useful only if NN batches are smaller than chunksize
                learner.train_on_batch(batch)
                if processed+skiprows >= cut:
                    break

    # collect trained models
    print("stopping training")
    modelfile = learner.stop_training(model_selection=False)[0]
    print("training stopped")

    ae = Recoder(loss_type=loss)
    ae.deserialize(modelfile)

    # randomly pick some samples for each class
    samples = {cl: [] for cl in classes}
    for chunk in pd.read_csv(set_filename, chunksize=chunksize, skiprows=cut,
                             header=None, delimiter=delimiter):

        batch = chunk.values
        for row, label in zip(batch, chunk[label_col].values):
            registered = samples.get(str(label))
            if registered is not None and len(registered) < sampling:
                registered.append(row)

    # train classifier
    from sklearn.svm import SVC
    y_train = []
    X_train = []
    for label, rows in samples.items():
        preprocessed = preproc.transform(np.vstack(rows))
        reduced = ae.reduce_dim(preprocessed, gpu)
        X_train.append(reduced)
        y_train += [label] * len(rows)
    classifier = SVC()
    classifier.fit(np.vstack(X_train), y_train)

    # collect predictions and reference labels
    y_pred = []
    y_true = []
    for chunk in pd.read_csv(set_filename, chunksize=8192, header=None,
                             delimiter=delimiter, skiprows=cut):
        relevant_data = chunk[chunk[label_col].isin(classes)]
        batch = relevant_data.values
        if len(batch) > 0:
            labels = relevant_data[label_col].values
            y_true += list(map(str, labels.tolist()))
            batch = preproc.transform(batch)
            reduced = ae.reduce_dim(batch, gpu)
            y_pred += classifier.predict(reduced).tolist()

    # reporting
    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    f1 = f1_score(y_true, y_pred, average='macro')
    print("F1[%s]: %f%%" % (control, 100 * f1))
