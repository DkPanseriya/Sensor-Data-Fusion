import numpy as np

from tools.detection_metrics import average_precision, box_iou, evaluate, match_image


def test_box_iou():
    assert np.isclose(box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0], 50 / 150)


def test_greedy_match_retries_second_best():
    # P1 (higher score) takes A; P2 overlaps A more than B but must still match the free B
    gt = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], float)
    preds = np.array([[0, 0, 10, 10], [3, 0, 13, 10]], float)
    tp = match_image(preds, [0.9, 0.8], gt, thresholds=[0.4])
    assert tp[:, 0].tolist() == [True, True]


def test_duplicates_are_false_positives():
    tp = match_image(np.array([[0, 0, 10, 10]] * 2, float), [0.9, 0.8], np.array([[0, 0, 10, 10]], float), [0.5])
    assert tp[:, 0].tolist() == [True, False]


def test_perfect_detector_scores_one():
    gt = {"a": (np.array([0, 1]), np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float))}
    pred = {"a": (np.array([0, 1]), gt["a"][1].copy(), np.array([0.9, 0.8]))}
    ap = evaluate(pred, gt, n_classes=3)
    assert np.allclose(ap[:2], 1.0) and np.isnan(ap[2]).all()


def test_missed_ground_truth_lowers_ap():
    tp = np.array([[True]])
    assert np.isclose(average_precision(tp, [0.9], n_gt=2)[0], 51 / 101)
