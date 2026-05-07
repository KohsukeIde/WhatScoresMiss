import numpy as np

from bds.class_pair import pairwise_balanced_accuracy, target_to_confusion_rate


def test_pairwise_balanced_accuracy_and_confusion_rate():
    features = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels = np.array([0, 0, 1, 1])
    assert pairwise_balanced_accuracy(features, labels, 0, 1) == 1.0

    pred = np.array([1, 1, 0, 1])
    target = np.array([0, 0, 0, 1])
    assert target_to_confusion_rate(pred, target, 0, 1) == 2 / 3
