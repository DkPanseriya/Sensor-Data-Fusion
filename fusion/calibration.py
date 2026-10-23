"""Sensor calibration of an INFRA-3DRC scene and the transforms between radar, image and ground."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _homogeneous(T):
    """3x4 extrinsic -> 4x4."""
    M = np.eye(4)
    M[:3] = np.asarray(T, float)
    return M


@dataclass
class Calibration:
    K: np.ndarray                  # 3x3 camera intrinsics
    dist: np.ndarray               # camera distortion coefficients
    radar_to_camera: np.ndarray    # 4x4
    radar_to_ground: np.ndarray    # 4x4, through the lidar frame
    image_to_ground: np.ndarray    # 3x3 homography, undistorted pixels -> ground (x forward, y left)

    @classmethod
    def from_file(cls, path):
        items = {c["calibration"]: c for c in json.loads(Path(path).read_text())["calibration"]}
        return cls(
            K=np.asarray(items["camera_01"]["k"], float),
            dist=np.asarray(items["camera_01"]["D"], float).ravel(),
            radar_to_camera=_homogeneous(items["radar_01_to_camera_01"]["T"]),
            radar_to_ground=_homogeneous(items["lidar_01_to_ground"]["T"]) @ _homogeneous(items["radar_01_to_lidar_01"]["T"]),
            image_to_ground=np.asarray(items["camera_01_to_ground_homography"]["T"], float),
        )

    def radar_to_image(self, points):
        """Radar points (N, 3) -> pixel coordinates (N, 2) in the raw, distorted image the detector sees."""
        points = np.asarray(points, float).reshape(-1, 3)
        if len(points) == 0:
            return np.empty((0, 2))
        rvec, _ = cv2.Rodrigues(self.radar_to_camera[:3, :3])
        pixels, _ = cv2.projectPoints(points, rvec, self.radar_to_camera[:3, 3], self.K, self.dist)
        return pixels.reshape(-1, 2)

    def radar_to_ground_xy(self, points):
        """Radar points (N, 3) -> ground-plane coordinates (N, 2), metres."""
        points = np.asarray(points, float).reshape(-1, 3)
        homog = np.hstack([points, np.ones((len(points), 1))])
        return (homog @ self.radar_to_ground.T)[:, :2]

    def image_to_ground_xy(self, pixels):
        """Pixels (N, 2) of points on the road -> ground-plane coordinates (N, 2), metres."""
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        if len(pixels) == 0:
            return np.empty((0, 2))
        undistorted = cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        homog = np.hstack([undistorted, np.ones((len(undistorted), 1))]) @ self.image_to_ground.T
        return homog[:, :2] / homog[:, 2:3]
