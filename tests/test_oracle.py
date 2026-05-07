import numpy as np

from bds.oracle import oracle_accuracy, oracle_predictions, topk_hit


def test_oracle_predictions_replace_only_when_target_in_topk():
    logits = np.array([[0.9, 0.8, 0.1], [0.9, 0.1, 0.8]])
    target = np.array([1, 1])
    pred = oracle_predictions(logits, target, k=2)
    assert pred.tolist() == [1, 0]
    assert topk_hit(logits, target, k=2).tolist() == [True, False]
    assert oracle_accuracy(logits, target, k=2) == 0.5
