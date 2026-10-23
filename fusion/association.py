"""Spatial association of camera boxes with radar objects.

Candidates come from the image plane: a radar object is a candidate for a box when its projected
centroid falls inside the box, expanded to absorb calibration and radar elevation error.
Ambiguities are resolved on the ground plane, by the distance between the radar object's reference
point and the box's bottom-centre.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Match:
    box: int
    objects: list            # indices into the radar object list (several when clusters were merged)
    case: str                # one-to-one, one-to-many, many-to-one, many-to-many


@dataclass
class AssociationResult:
    matches: list = field(default_factory=list)
    unmatched_boxes: list = field(default_factory=list)


def expand_boxes(boxes, scale=1.2):
    """Scale xyxy boxes (N, 4) about their centres."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) * scale / 2
    return np.hstack([centre - half, centre + half])


def candidate_matrix(points_uv, boxes, scale=1.2):
    """Boolean (objects, boxes): True where the object's pixel lies inside the expanded box."""
    points_uv = np.asarray(points_uv, float).reshape(-1, 2)
    ex = expand_boxes(boxes, scale)
    u, v = points_uv[:, 0:1], points_uv[:, 1:2]
    return (u > ex[None, :, 0]) & (u < ex[None, :, 2]) & (v > ex[None, :, 1]) & (v < ex[None, :, 3])


def _components(matrix):
    """Connected components of the bipartite object-box graph: list of (object_ids, box_ids)."""
    n_obj, n_box = matrix.shape
    parent = list(range(n_obj + n_box))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for o, b in zip(*np.nonzero(matrix)):
        parent[find(o)] = find(n_obj + b)

    groups = {}
    for node in range(n_obj + n_box):
        if node < n_obj and not matrix[node].any():
            continue
        if node >= n_obj and not matrix[:, node - n_obj].any():
            continue
        groups.setdefault(find(node), ([], []))[0 if node < n_obj else 1].append(node if node < n_obj else node - n_obj)
    return list(groups.values())


def merge_groups(object_ids, ground_xy, doppler, max_doppler_gap=0.75, max_lateral_gap=2.0):
    """Group objects that plausibly belong to one road user: similar Doppler and lateral position."""
    parent = {i: i for i in object_ids}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a in object_ids:
        for b in object_ids:
            if a < b and abs(doppler[a] - doppler[b]) < max_doppler_gap \
                    and abs(ground_xy[a][1] - ground_xy[b][1]) < max_lateral_gap:
                parent[find(a)] = find(b)

    groups = {}
    for i in object_ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def associate(boxes, box_ground, object_uv, object_ground, object_doppler, allowed_boxes=None,
              scale=1.2, max_doppler_gap=0.75, max_lateral_gap=2.0):
    """Associate boxes with radar objects.

    boxes:          (B, 4) xyxy pixels
    box_ground:     (B, 2) ground position of each box's bottom-centre
    object_uv:      (O, 2) projected radar centroids, pixels
    object_ground:  (O, 2) ground position of each object's reference point
    object_doppler: (O,)   mean range rate
    allowed_boxes:  restrict to these boxes (the second pass only looks at boxes the first pass left)
    """
    n_boxes = len(boxes)
    allowed = list(range(n_boxes)) if allowed_boxes is None else sorted(allowed_boxes)
    result = AssociationResult()
    if not allowed:
        return result
    if len(object_uv) == 0:
        result.unmatched_boxes = allowed
        return result

    matrix = candidate_matrix(object_uv, boxes, scale)
    mask = np.zeros(n_boxes, bool)
    mask[allowed] = True
    matrix &= mask[None, :]

    def dist(o, b):
        return float(np.linalg.norm(np.asarray(object_ground[o]) - np.asarray(box_ground[b])))

    matched = set()
    for objs, bxs in _components(matrix):
        if len(objs) == 1 and len(bxs) == 1:
            result.matches.append(Match(bxs[0], objs, "one-to-one"))
            matched.add(bxs[0])
        elif len(bxs) == 1:
            box = bxs[0]
            groups = merge_groups(objs, object_ground, object_doppler, max_doppler_gap, max_lateral_gap)
            best = min(groups, key=lambda g: min(dist(o, box) for o in g))
            result.matches.append(Match(box, sorted(best), "many-to-one"))
            matched.add(box)
        elif len(objs) == 1:
            obj = objs[0]
            box = min(bxs, key=lambda b: dist(obj, b))
            result.matches.append(Match(box, objs, "one-to-many"))
            matched.add(box)
        else:
            cost = np.array([[dist(o, b) if matrix[o, b] else 1e6 for b in bxs] for o in objs])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e6:
                    result.matches.append(Match(bxs[c], [objs[r]], "many-to-many"))
                    matched.add(bxs[c])

    result.matches.sort(key=lambda m: m.box)
    result.unmatched_boxes = [b for b in allowed if b not in matched]
    return result
