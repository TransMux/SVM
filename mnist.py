import numpy as np
from scipy.special import comb
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from collections import Counter

from model.MutiClassSVM import MultiClassSVM


def measure_performance(y_true, y_pred, average="macro"):
    measures = defaultdict(list)
    measures["Accuracy"].append(round(accuracy_score(y_true, y_pred), 4))
    measures["Precision"].append(round(precision_score(y_true, y_pred, average=average), 4))
    measures["Recall"].append(round(recall_score(y_true, y_pred, average=average), 4))
    measures["F1"].append(round(f1_score(y_true, y_pred, average=average), 4))
    measures["MCC"].append(round(matthews_corrcoef(y_true, y_pred), 4))
    return pd.DataFrame.from_dict(measures)


data = np.load('mnist.npz')
x_train = data['a']
y_train = data['b']
x_test = data['c']
y_test = data['d']
print("Number of training samples:       ", len(y_train))
print("Distribution of training samples: ", Counter(y_train))
print("Number of test samples:           ", len(y_test))
print("Distribution of training samples: ", Counter(y_test))


def label2binary(y, label):
    """
    Map label val to +1 and the other labels to -1.

    Paramters:
    ----------
    y : `numpy.ndarray`
        (nData,) The labels of two classes.
    val : `int`
        The label to map to +1.

    Returns:
    --------
    y : `numpy.ndarray`
        (nData,) Maps the val label to +1 and the other label to -1.
    """
    return (2 * (y == label).astype(int)) - 1


def split4ovo(X, y):
    """
    Split the data into kchoose2 datasets

    Paramters:
    ----------
    X : `numpy.ndarray`
        (nData, nDim) The training data.
    y : `numpy.ndarray`
        (nData,) Corresponding training labels.

    Returns:
    --------
    X_ovo : `dict` of `numpy.ndarray`
        Dictionary of datasets for one vs one mutliclass classification.
    y_ovo : `dict` of `numpy.ndarray`
        Dictionary of corresponding labels for one vs one mutliclass classification.
    """
    X_ovo = dict()
    y_ovo = dict()
    unique_classes = np.unique(y).astype(int)
    nClasses = len(unique_classes)
    for i in unique_classes:
        unique_classes = np.delete(unique_classes, np.where(unique_classes == i), axis=0)
        for j in unique_classes:
            # Get indices for which labels are i vs j
            indices = np.logical_or(y == i, y == j)
            # Get trainxs from indices
            X_ovo[str(i) + "vs" + str(j)] = X[indices]
            # Get corresponding labels, i->1 and j->-1
            y_ovo[str(i) + "vs" + str(j)] = label2binary(y[indices], i)
    return X_ovo, y_ovo


def split4ovr(y):
    """
    Change the labels such that we map the class label that is the
    one in one vs all to +1 and the all to -1.

    Paramters:
    ----------
    y : `numpy.ndarray`
        (nData,) Corresponding training labels.

    Returns:
    --------
    y_ovr : `dict` of `numpy.ndarray`
        Dictionary of labels for one vs rest mutliclass classification.
    """
    unique_classes = np.unique(y).astype(int)
    y_ovr = dict()
    for i in unique_classes:
        y_ovr[str(i) + "vsR"] = label2binary(y, i)
    return y_ovr


x_train_ovo, y_train_ovo = split4ovo(x_train, y_train)
# Initialise L1-SVM
L1_barrier_svm = MultiClassSVM(C=1.0,
                               kernel="gauss",
                               param='scale',
                               decision_function_shape="ovo",
                               loss_fn='L1',
                               opt_algo="barrier")

# Barrier fit
L1_barrier_svm.fit(x_train_ovo, y_train_ovo,
                   t=1,
                   mu=20,
                   tol=1e-6,
                   max_iter=100,
                   tolNewton=1e-12,
                   maxIterNewton=100)

# Test
L1_barrier_yhat = L1_barrier_svm.predict(x_test)
print("Time taken: ", L1_barrier_svm.time_taken)
measure_performance(y_test, L1_barrier_yhat, average="macro")
