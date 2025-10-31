import numpy as np
from typing import Callable, Sequence

from .metrics import euclidean
from .postprocess import majority_label, average_label


def knn_predict(
    X_train: np.ndarray,
    y_train: Sequence,
    X_test: np.ndarray,
    k: int = 3,
    distance: Callable = euclidean,
    classify: bool = True,
):
    """
    KNN for classification (majority vote) or regression (average).
    """
    X_train = np.asarray(X_train, float)
    X_test  = np.asarray(X_test,  float)
    y_train = np.asarray(y_train)

    assert 1 <= k <= len(X_train), "k must be between 1 and len(X_train)"
    preds = []
    for x in X_test:
        dists = np.array([distance(x, xi) for xi in X_train], float)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        preds.append(majority_label(nn_labels) if classify else average_label(nn_labels))
    return np.array(preds, dtype=object if classify else float)
