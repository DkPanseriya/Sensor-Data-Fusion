"""Range-adaptive DBSCAN for sparse automotive radar returns."""
from dataclasses import dataclass

import numpy as np


@dataclass
class RadarObject:
    """A radar cluster, or a single noise point treated as an object candidate."""
    centroid: np.ndarray      # (3,) radar frame, metres
    reference: np.ndarray     # (3,) nearest-range x and z, mean lateral y: the point compared with the box bottom
    doppler: float            # mean range rate, m/s
    points: np.ndarray        # (M, 3) member points


def filter_moving(cloud, max_range=100.0, min_speed=0.1):
    """Keep returns closer than max_range with |range rate| above min_speed (moving road users)."""
    keep = (cloud["range"] < max_range) & (np.abs(cloud["range_rate"]) > min_speed)
    return cloud[keep]


def neighbourhood(xy, ranges, eps=(0.1, 0.25), offset=1.5):
    """Boolean (N, N) neighbour matrix.

    Returns get sparser with distance, so the neighbourhood is an ellipse whose semi-axes grow with
    range: eps_i * sqrt(r) + offset, with r the mean range of the two points (symmetric by construction).
    """
    r = (ranges[:, None] + ranges[None, :]) / 2
    a = eps[0] * np.sqrt(r) + offset
    b = eps[1] * np.sqrt(r) + offset
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    near = (dx / a) ** 2 + (dy / b) ** 2 < 1
    np.fill_diagonal(near, False)
    return near


def dbscan(near, min_neighbours=2):
    """Labels (N,) from a neighbour matrix: -1 noise, 0..k-1 clusters.

    A point is a core point when it has at least `min_neighbours` neighbours, not counting itself.
    """
    n = len(near)
    labels = np.full(n, -1)
    core = near.sum(axis=1) >= min_neighbours
    cluster = 0
    for seed in np.flatnonzero(core):
        if labels[seed] != -1:
            continue
        labels[seed] = cluster
        frontier = [seed]
        while frontier:
            point = frontier.pop()
            if not core[point]:
                continue
            for neighbour in np.flatnonzero(near[point]):
                if labels[neighbour] == -1:
                    labels[neighbour] = cluster
                    frontier.append(neighbour)
        cluster += 1
    return labels


def cluster_radar(cloud, eps=(0.1, 0.25), offset=1.5, min_neighbours=2):
    """Cluster a radar scan. Returns (clusters, noise), both lists of RadarObject."""
    cloud = filter_moving(cloud)
    if len(cloud) == 0:
        return [], []
    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(float)
    ranges = cloud["range"].astype(float)
    speeds = cloud["range_rate"].astype(float)
    labels = dbscan(neighbourhood(xyz[:, :2], ranges, eps, offset), min_neighbours)

    clusters = []
    for label in range(labels.max() + 1):
        members = labels == label
        pts = xyz[members]
        nearest = pts[np.argmin(ranges[members])]
        clusters.append(RadarObject(
            centroid=pts.mean(axis=0),
            reference=np.array([nearest[0], pts[:, 1].mean(), nearest[2]]),
            doppler=float(speeds[members].mean()),
            points=pts,
        ))
    noise = [RadarObject(centroid=p, reference=p, doppler=float(v), points=p[None])
             for p, v in zip(xyz[labels == -1], speeds[labels == -1])]
    return clusters, noise
