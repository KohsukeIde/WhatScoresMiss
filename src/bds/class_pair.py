"""Class-pair diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def pairwise_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    class_a: int,
    class_b: int,
    seed: int = 0,
) -> float:
    """Fit a simple logistic probe and return balanced accuracy."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    mask = np.logical_or(labels == class_a, labels == class_b)
    x = features[mask]
    y = labels[mask]
    if len(np.unique(y)) != 2:
        raise ValueError("both classes must be present")
    y_bin = (y == class_b).astype(int)
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(x, y_bin)
    pred = clf.predict(x)
    return float(balanced_accuracy_score(y_bin, pred))


def target_to_confusion_rate(
    pred: np.ndarray, target: np.ndarray, target_class: int, confusion_class: int
) -> float:
    """Rate at which target_class points/samples are predicted as confusion_class."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    mask = target == target_class
    denom = mask.sum()
    if denom == 0:
        return float("nan")
    return float(np.mean(pred[mask] == confusion_class))
