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

def hinge_loss(y_true, pred_decision, labels=None, sample_weight=None):
    check_consistent_length(y_true, pred_decision, sample_weight)
    pred_decision = check_array(pred_decision, ensure_2d=False)
    y_true = column_or_1d(y_true)
    y_true_unique = np.unique(y_true)
    if y_true_unique.size > 2:
        if (labels is None and pred_decision.ndim > 1 and
                (np.size(y_true_unique) != pred_decision.shape[1])):
            raise ValueError("Please include all labels in y_true "
                             "or pass labels as third argument")
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = np.ones_like(pred_decision, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        margin = pred_decision[~mask]
        margin -= np.max(pred_decision[mask].reshape(y_true.shape[0], -1),
                         axis=1)
    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        pred_decision = column_or_1d(pred_decision)
        pred_decision = np.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    losses[losses <= 0] = 0
    return losses

