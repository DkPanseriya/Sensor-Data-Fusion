"""Multi-object tracking of fused objects on the ground plane with an extended Kalman filter.

State per track: x = (px, py, vx, vy) in the ground frame, constant-velocity motion.

Measurements:
- Fused object (camera box + radar): the radar's own quantities, z = (range, azimuth, range rate),
  measured in the radar frame. They relate to the state through a nonlinear h(x), linearised at
  each update with its Jacobian, which makes this an extended Kalman filter. The range rate
  (Doppler) observes velocity directly, so a track knows its speed from the first radar frame.
- Camera-only object: its ground position from the homography, a linear measurement with a
  larger noise.

Detections are assigned to tracks by the Hungarian algorithm on the Mahalanobis distance of the
ground position, gated at the 99% chi-square bound.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE = 9.21          # chi-square, 2 degrees of freedom, 99%
H_POS = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


@dataclass
class Track:
    id: int
    x: np.ndarray                 # (4,) px, py, vx, vy
    P: np.ndarray                 # (4, 4)
    label: str
    hits: int = 1
    misses: int = 0
    history: list = field(default_factory=list)

    @property
    def position(self):
        return self.x[:2]

    @property
    def velocity(self):
        return self.x[2:]


class RadarGeometry:
    """2D pose of the radar on the ground plane, to express ground states in radar coordinates."""

    def __init__(self, radar_to_ground):
        self.R = np.asarray(radar_to_ground, float)[:2, :2]      # radar -> ground rotation (yaw)
        self.t = np.asarray(radar_to_ground, float)[:2, 3]       # radar position on the ground

    def h(self, x, dz):
        """Predicted (range, azimuth, range rate) of state x, for a target dz metres from the radar's height."""
        p = self.R.T @ (x[:2] - self.t)
        v = self.R.T @ x[2:]
        rho2 = p @ p
        r3 = np.sqrt(rho2 + dz ** 2)                              # radar range is 3D
        z = np.array([r3, np.arctan2(p[1], p[0]), (p @ v) / r3])

        # Jacobian with respect to radar-frame position and velocity, chained to the ground frame
        dr_dp = p / r3
        dth_dp = np.array([-p[1], p[0]]) / rho2
        drr_dp = v / r3 - (p @ v) * p / r3 ** 3
        drr_dv = p / r3
        J = np.zeros((3, 4))
        J[0, :2] = dr_dp @ self.R.T
        J[1, :2] = dth_dp @ self.R.T
        J[2, :2] = drr_dp @ self.R.T
        J[2, 2:] = drr_dv @ self.R.T
        return z, J


def radar_measurement(obj):
    """(range, azimuth, range rate) and height offset of a fused object's radar reference point."""
    ref = np.mean([r.reference for r in obj.radar], axis=0)        # radar frame, metres
    return np.array([np.linalg.norm(ref), np.arctan2(ref[1], ref[0]), obj.doppler]), ref[2]


class Tracker:
    def __init__(self, radar_to_ground, accel_noise=3.0, range_sigma=0.5, azimuth_sigma_deg=1.0,
                 range_rate_sigma=0.3, position_sigma=0.5, camera_sigma=2.5, min_hits=3, max_misses=5,
                 history=30):
        self.geometry = RadarGeometry(radar_to_ground)
        self.q = accel_noise ** 2
        self.R_radar = np.diag([range_sigma ** 2, np.radians(azimuth_sigma_deg) ** 2, range_rate_sigma ** 2])
        self.r_assoc = {"radar": position_sigma ** 2, "camera": camera_sigma ** 2}
        self.min_hits, self.max_misses, self.history = min_hits, max_misses, history
        self.tracks, self.next_id, self.time = [], 1, None

    def _predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = self.q * G @ G.T
        for t in self.tracks:
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q

    @staticmethod
    def _correct(t, innovation, J, R):
        S = J @ t.P @ J.T + R
        K = t.P @ J.T @ np.linalg.inv(S)
        t.x = t.x + K @ innovation
        I_KJ = np.eye(4) - K @ J
        t.P = I_KJ @ t.P @ I_KJ.T + K @ R @ K.T                   # Joseph form keeps P symmetric

    def _update(self, t, obj, position):
        if obj.radar:
            z, dz = radar_measurement(obj)
            z_pred, J = self.geometry.h(t.x, dz)
            innovation = z - z_pred
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi
            self._correct(t, innovation, J, self.R_radar)
        else:
            self._correct(t, position - t.position, H_POS, self.r_assoc["camera"] * np.eye(2))

    def _new_track(self, obj, position, r):
        x = np.array([*position, 0.0, 0.0])
        P = np.diag([r, r, 100.0, 100.0])                          # velocity unknown: sigma 10 m/s
        if obj.radar:
            # Doppler fixes the radial component of the velocity; the tangential one stays unknown
            radial = position - self.geometry.t
            u = radial / np.linalg.norm(radial)
            (r3, _, rate), dz = radar_measurement(obj)
            rho = max(np.sqrt(max(r3 ** 2 - dz ** 2, 0.0)), 0.5)
            x[2:] = rate * r3 / rho * u                         # 3D range rate -> horizontal radial speed
            rot = np.array([[u[0], -u[1]], [u[1], u[0]]])
            P[2:, 2:] = rot @ np.diag([0.5 ** 2, 100.0]) @ rot.T
        t = Track(self.next_id, x, P, obj.label)
        self.next_id += 1
        self.tracks.append(t)
        return t

    def update(self, objects, time):
        """Assign `track_id` (None until a track is confirmed), `velocity` and `trail` to each object."""
        dt = 0.1 if self.time is None else max(time - self.time, 1e-3)
        self.time = time
        self._predict(dt)

        z = [o.radar_ground if o.radar_ground is not None else o.ground for o in objects]
        r = [self.r_assoc["radar"] if o.radar_ground is not None else self.r_assoc["camera"] for o in objects]

        cost = np.full((len(self.tracks), len(objects)), 1e6)
        for i, t in enumerate(self.tracks):
            for j, (zj, rj) in enumerate(zip(z, r)):
                S = H_POS @ t.P @ H_POS.T + rj * np.eye(2)
                d = np.asarray(zj) - t.position
                m = float(d @ np.linalg.solve(S, d))
                if m < GATE:
                    # negative log-likelihood: without log|S| a vague camera-only detection
                    # would beat a precise radar detection at a similar distance
                    cost[i, j] = m + np.log(np.linalg.det(S))
        pairs = [(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6] if cost.size else []

        track_of = [None] * len(objects)
        for i, j in pairs:
            t = self.tracks[i]
            self._update(t, objects[j], np.asarray(z[j], float))
            t.hits += 1
            t.misses = 0
            t.label = objects[j].label
            track_of[j] = t

        assigned = {i for i, _ in pairs}
        for i, t in enumerate(self.tracks):
            if i not in assigned:
                t.misses += 1
        # a confirmed track survives a few missed frames, a tentative one none
        self.tracks = [t for t in self.tracks if t.misses <= (self.max_misses if t.hits >= self.min_hits else 0)]

        for j, o in enumerate(objects):
            if track_of[j] is None:
                track_of[j] = self._new_track(o, np.asarray(z[j], float), r[j])

        for o, t in zip(objects, track_of):
            t.history = (t.history + [t.position.copy()])[-self.history:]
            confirmed = t.hits >= self.min_hits
            o.track_id = t.id if confirmed else None
            o.velocity = t.velocity.copy() if confirmed else None
            o.trail = list(t.history) if confirmed else []
        return objects
