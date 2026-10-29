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


def test_unique_retained_shape_part_iou():
    from bds.retained_scoring import unique_retained_shape_part_iou

    # Duplicate point 7 is averaged. Part 0 has the largest logit but is not
    # valid for this category. Part 3 is absent and contributes IoU 1.
    logits = np.array([[20, 4, 0, 0], [20, 2, 0, 0], [20, 3, 2, 0]], dtype=float)
    target = np.array([1, 1, 2])
    original = np.array([7, 7, 9])
    assert unique_retained_shape_part_iou(logits, target, original, np.array([1, 2, 3])) == 0.5


def test_unique_retained_shape_part_iou_rejects_conflicting_labels():
    import pytest
    from bds.retained_scoring import unique_retained_shape_part_iou

    with pytest.raises(ValueError, match="inconsistent targets"):
        unique_retained_shape_part_iou(
            np.ones((2, 3)), np.array([0, 1]), np.array([4, 4]), np.array([0, 1])
        )
