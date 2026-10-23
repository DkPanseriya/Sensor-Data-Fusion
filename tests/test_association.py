import numpy as np

from fusion.association import associate, candidate_matrix, expand_boxes, merge_groups


def test_expand_boxes_about_centre():
    assert np.allclose(expand_boxes([[10, 10, 20, 30]], 1.2), [[9, 8, 21, 32]])


def test_candidate_matrix():
    m = candidate_matrix([[15, 20], [100, 100]], [[10, 10, 20, 30]])
    assert m.tolist() == [[True], [False]]


def run(boxes, box_ground, uv, ground, doppler, **kw):
    return associate(np.array(boxes, float), np.array(box_ground, float), np.array(uv, float),
                     np.array(ground, float), np.array(doppler, float), **kw)


def test_no_radar_keeps_every_box():
    r = associate(np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float), np.zeros((2, 2)),
                  np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    assert r.matches == [] and r.unmatched_boxes == [0, 1]


def test_one_to_one_and_unmatched():
    r = run([[0, 0, 10, 10], [50, 0, 60, 10]], [[10, 0], [10, 5]], [[5, 5]], [[10, 0]], [1.0])
    assert [(m.box, m.objects, m.case) for m in r.matches] == [(0, [0], "one-to-one")]
    assert r.unmatched_boxes == [1]


def test_one_to_many_picks_nearest_box_on_ground():
    # one radar object inside two overlapping boxes; box 1 is closer on the ground
    r = run([[0, 0, 20, 20], [5, 5, 25, 25]], [[30, 0], [12, 0]], [[12, 12]], [[11, 0]], [2.0])
    assert [(m.box, m.case) for m in r.matches] == [(1, "one-to-many")]
    assert r.unmatched_boxes == [0]


def test_many_to_one_keeps_a_lone_cluster_that_is_closest():
    # two far clusters merge into a group, a single cluster sits right at the box: the single one wins
    r = run([[0, 0, 40, 40]], [[10, 0]], [[10, 10], [12, 12], [30, 30]],
            [[30, 5], [31, 5], [10.2, 0]], [5.0, 5.1, 2.0])
    assert r.matches[0].objects == [2]


def test_many_to_one_merges_parts_of_one_vehicle():
    r = run([[0, 0, 40, 40]], [[10, 0]], [[10, 10], [20, 20]], [[10, 0.5], [13, 0.0]], [5.0, 5.2])
    assert r.matches[0].objects == [0, 1] and r.matches[0].case == "many-to-one"


def test_many_to_many_loses_no_box():
    # cluster 0 inside boxes A and B, cluster 1 inside A only: both boxes must be resolved
    boxes = [[0, 0, 30, 30], [20, 0, 50, 30]]
    r = run(boxes, [[10, 0], [10, 4]], [[25, 15], [5, 15]], [[10, 4], [10, 0]], [1.0, 3.0])
    assert sorted(m.box for m in r.matches) == [0, 1]
    assert r.unmatched_boxes == []


def test_second_pass_only_uses_allowed_boxes():
    r = run([[0, 0, 10, 10], [20, 0, 30, 10]], [[10, 0], [10, 5]], [[5, 5], [25, 5]], [[10, 0], [10, 5]],
            [1.0, 1.0], allowed_boxes=[1])
    assert [m.box for m in r.matches] == [1] and r.unmatched_boxes == []


def test_merge_groups_gates_on_doppler_and_lateral():
    ground = {0: [10, 0], 1: [12, 1], 2: [12, 6]}
    doppler = {0: 5.0, 1: 5.3, 2: 5.0}
    assert sorted(map(sorted, merge_groups([0, 1, 2], ground, doppler))) == [[0, 1], [2]]
