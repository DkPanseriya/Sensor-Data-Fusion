import numpy as np

from fusion.clustering import cluster_radar, dbscan, neighbourhood


def cloud(points):
    dtype = [("range", "f4"), ("range_rate", "f4"), ("x", "f4"), ("y", "f4"), ("z", "f4")]
    rows = [(np.hypot(x, y), v, x, y, 0.0) for x, y, v in points]
    return np.array(rows, dtype=dtype)


def test_neighbourhood_is_symmetric():
    xy = np.array([[10, 0], [11.4, 0], [60, 0]], float)
    near = neighbourhood(xy, np.linalg.norm(xy, axis=1))
    assert (near == near.T).all() and near[0, 1] and not near[0, 2]


def test_dbscan_min_neighbours_excludes_self():
    near = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], bool)
    assert (dbscan(near, min_neighbours=2) == 0).all()
    assert (dbscan(near, min_neighbours=3) == -1).all()


def test_two_objects_and_static_clutter():
    scan = cloud([(20, 0, 5), (20.5, 0.3, 5), (21, -0.2, 5),      # car
                  (40, 8, -3), (40.4, 8.2, -3), (40.8, 8.1, -3),  # second car
                  (30, -10, 0.0),                                 # static return, filtered out
                  (70, 20, 1.0)])                                 # lone moving return: noise
    clusters, noise = cluster_radar(scan)
    assert len(clusters) == 2 and len(noise) == 1
    assert np.isclose(clusters[0].doppler, 5)
    # reference point: x of the nearest return, mean lateral position
    assert np.isclose(clusters[0].reference[0], 20) and np.isclose(clusters[0].reference[1], 0.1 / 3 * 1, atol=1e-6)


def test_empty_scan():
    assert cluster_radar(cloud([])) == ([], [])
