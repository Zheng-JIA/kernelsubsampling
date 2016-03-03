from __future__ import division

import warnings
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.spatial.distance import hamming as sp_hamming

from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.utils.fixes import bincount

from sklearn.metrics.base import UndefinedMetricWarning


def log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None):
    lb = LabelBinarizer()
    T = lb.fit_transform(y_true)
    if T.shape[1] == 1:
        T = np.append(1 - T, T, axis=1)

    # Clipping
    Y = np.clip(y_pred, eps, 1 - eps)

    # This happens in cases when elements in y_pred have type "str".
    if not isinstance(Y, np.ndarray):
        raise ValueError("y_pred should be an array of floats.")

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)
    # Check if dimensions are consistent.
    check_consistent_length(T, Y)
    T = check_array(T)
    Y = check_array(Y)
    if T.shape[1] != Y.shape[1]:
        raise ValueError("y_true and y_pred have different number of classes "
                         "%d, %d" % (T.shape[1], Y.shape[1]))

    # Renormalize
    Y /= Y.sum(axis=1)[:, np.newaxis]
    loss = -(T * np.log(Y)).sum(axis=1)

    return loss 
