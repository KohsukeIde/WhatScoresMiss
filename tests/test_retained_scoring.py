import numpy as np

from bds.retained_scoring import accuracy, average_duplicate_logits, mean_iou


def test_average_duplicate_logits():
    logits = np.array([[1.0, 3.0], [3.0, 1.0], [5.0, 7.0]])
    original = np.array([4, 4, 8])
    unique, avg = average_duplicate_logits(logits, original)
    assert unique.tolist() == [4, 8]
    np.testing.assert_allclose(avg, np.array([[2.0, 2.0], [5.0, 7.0]]))


def test_mean_iou_and_accuracy():
    pred = np.array([0, 1, 1, 2])
    target = np.array([0, 1, 2, 2])
    assert accuracy(pred, target) == 0.75
    assert mean_iou(pred, target, num_classes=3) == (1.0 + 0.5 + 0.5) / 3.0
