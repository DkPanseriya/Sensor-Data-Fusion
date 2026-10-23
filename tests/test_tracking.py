import numpy as np

from fusion.clustering import RadarObject
from fusion.pipeline import FusedObject
from fusion.tracking import RadarGeometry, Tracker

RADAR_HEIGHT = 3.5


def obj(x, y, vx=0.0, vy=0.0, radar=True):
    """A fused object at ground (x, y) moving with (vx, vy); the radar sits at the ground origin."""
    o = FusedObject(box=np.zeros(4), label="car", confidence=0.9, ground=np.array([x + 2.0, y]))
    if radar:
        ref = np.array([x, y, -RADAR_HEIGHT])
        doppler = (x * vx + y * vy) / np.linalg.norm(ref)
        o.radar = [RadarObject(centroid=ref, reference=ref, doppler=doppler, points=ref[None])]
        o.radar_ground = np.array([x, y], float)
    return o


def run(tracker, frames, dt=0.1):
    return [tracker.update([obj(*p) for p in frame], k * dt) for k, frame in enumerate(frames)]


def moving(x0, y0, vx, vy, n, dt=0.1):
    return [(x0 + vx * k * dt, y0 + vy * k * dt, vx, vy) for k in range(n)]


def test_jacobian_matches_finite_differences():
    rot = np.eye(4)
    rot[:2, :2] = [[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]]
    rot[:2, 3] = [1.0, -2.0]
    g = RadarGeometry(rot)
    x = np.array([25.0, 6.0, -8.0, 3.0])
    _, J = g.h(x, -3.5)
    eps = 1e-6
    numeric = np.stack([(g.h(x + eps * e, -3.5)[0] - g.h(x - eps * e, -3.5)[0]) / (2 * eps) for e in np.eye(4)], axis=1)
    assert np.allclose(J, numeric, atol=1e-6)


def test_track_confirmed_after_three_hits_and_keeps_its_id():
    frames = run(Tracker(np.eye(4)), [[p] for p in moving(20, 0, 15, 0, 6)])
    assert [f[0].track_id for f in frames] == [None, None, 1, 1, 1, 1]


def test_doppler_gives_radial_speed_from_the_first_frame():
    t = Tracker(np.eye(4))
    t.update([obj(30, 0, 12, 0)], 0.0)
    assert np.allclose(t.tracks[0].velocity, [12, 0], atol=0.6)


def test_velocity_converges_for_constant_motion():
    frames = run(Tracker(np.eye(4)), [[p] for p in moving(20, 5, 15, -2, 25)])
    assert np.allclose(frames[-1][0].velocity, [15, -2], atol=0.5)


def test_crossing_objects_keep_their_ids():
    a, b = moving(10, 2, 12, 0, 25), moving(40, -2, -12, 0, 25)
    frames = run(Tracker(np.eye(4)), [[pa, pb] for pa, pb in zip(a, b)])
    first = [o.track_id for o in frames[4]]
    last = [o.track_id for o in frames[-1]]
    assert first == last and len(set(first)) == 2


def test_track_survives_short_gap_and_is_dropped_after_long_one():
    t = Tracker(np.eye(4), max_misses=3)
    run(t, [[p] for p in moving(20, 0, 15, 0, 5)])
    t.update([], 0.5)
    t.update([], 0.6)
    assert t.update([obj(20 + 15 * 0.7, 0, 15, 0)], 0.7)[0].track_id == 1      # bridged two missed frames
    for k in range(8, 13):
        t.update([], k * 0.1)
    assert t.tracks == []


def test_camera_only_object_is_tracked():
    t = Tracker(np.eye(4))
    frames = [t.update([obj(20 + 1.5 * k, 0, radar=False)], k * 0.1) for k in range(5)]
    assert frames[-1][0].track_id == 1


def test_vague_camera_detection_does_not_steal_a_radar_track():
    t = Tracker(np.eye(4))
    run(t, [[p] for p in moving(10, 0, 0.001, 0, 5)])
    near_radar = obj(11.0, 0)                               # 1 m off, radar-backed
    camera = obj(13.5, 0, radar=False)                      # 3.5 m off, camera only
    camera.ground = np.array([13.5, 0.0])
    out = t.update([camera, near_radar], 0.5)
    assert out[1].track_id == 1
