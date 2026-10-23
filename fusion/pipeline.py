"""Per-frame late fusion: detect, cluster, project, associate."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .association import associate
from .calibration import Calibration
from .clustering import cluster_radar
from .pcd import read_pcd


@dataclass
class FusedObject:
    box: np.ndarray           # (4,) xyxy pixels
    label: str
    confidence: float
    ground: np.ndarray        # (2,) ground position of the box bottom-centre, metres
    radar: list = field(default_factory=list)   # RadarObjects backing this detection
    source: str = "none"      # "cluster", "noise" or "none"
    case: str = ""

    @property
    def range_m(self):
        return float(np.linalg.norm(self.radar[0].reference[:2])) if self.radar else None

    @property
    def doppler(self):
        return float(np.mean([r.doppler for r in self.radar])) if self.radar else None


@dataclass
class FrameResult:
    image: Path
    objects: list
    clusters: list
    noise: list
    calibration: Calibration


def _timestamp(path):
    # camera_01__2023-06-02-21-28-09-314.png -> datetime with milliseconds
    stamp = path.stem.split("__")[-1]
    return datetime.strptime(stamp[:-4], "%Y-%m-%d-%H-%M-%S").timestamp() + int(stamp[-3:]) / 1000


def scene_frames(scene_dir, max_offset_s=0.05):
    """Pair each camera frame of a scene with the radar scan closest in time."""
    scene_dir = Path(scene_dir)
    images = sorted((scene_dir / "camera_01" / "camera_01__data").glob("*.png"))
    scans = sorted((scene_dir / "radar_01" / "radar_01__data").glob("*.pcd"))
    if not images or not scans:
        raise FileNotFoundError(f"{scene_dir}: expected camera_01/camera_01__data/*.png and radar_01/radar_01__data/*.pcd")
    scan_times = np.array([_timestamp(s) for s in scans])
    pairs = []
    for image in images:
        i = int(np.argmin(np.abs(scan_times - _timestamp(image))))
        if abs(scan_times[i] - _timestamp(image)) <= max_offset_s:
            pairs.append((image, scans[i]))
    return pairs


class FusionPipeline:
    def __init__(self, model, calibration, conf=0.5, eps=(0.1, 0.25), min_neighbours=2, box_scale=1.2):
        self.model = model
        self.calibration = calibration
        self.conf = conf
        self.eps = eps
        self.min_neighbours = min_neighbours
        self.box_scale = box_scale

    def detect(self, image):
        result = self.model.predict(str(image), conf=self.conf, verbose=False)[0]
        boxes = result.boxes
        names = [self.model.names[int(c)] for c in boxes.cls.cpu().numpy()]
        return boxes.xyxy.cpu().numpy(), names, boxes.conf.cpu().numpy()

    def __call__(self, image, scan):
        cal = self.calibration
        xyxy, names, confs = self.detect(image)
        bottom_centre = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1) if len(xyxy) else np.empty((0, 2))
        box_ground = cal.image_to_ground_xy(bottom_centre)
        objects = [FusedObject(box=b, label=n, confidence=float(c), ground=g)
                   for b, n, c, g in zip(xyxy, names, confs, box_ground)]

        clusters, noise = cluster_radar(read_pcd(scan), self.eps, min_neighbours=self.min_neighbours)

        def run(candidates, source, allowed):
            if not candidates:
                return list(allowed)
            uv = cal.radar_to_image([c.centroid for c in candidates])
            ground = cal.radar_to_ground_xy([c.reference for c in candidates])
            doppler = np.array([c.doppler for c in candidates])
            result = associate(xyxy, box_ground, uv, ground, doppler, allowed_boxes=allowed, scale=self.box_scale)
            for m in result.matches:
                objects[m.box].radar = [candidates[i] for i in m.objects]
                objects[m.box].source = source
                objects[m.box].case = m.case
            return result.unmatched_boxes

        left = run(clusters, "cluster", range(len(objects)))
        run(noise, "noise", left)
        return FrameResult(Path(image), objects, clusters, noise, cal)
