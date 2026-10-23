"""Score association and tracking against the INFRA-3DRC labels.

The dataset gives every camera box and every labelled radar point a track_id, the same id for the same
road user in both sensors. That allows two scores:

- Association: for each camera box, did fusion attach the radar points of the same road user?
- Tracking: do the tracker's ids follow the labelled ids (ID switches, IDF1)?

    python tools/evaluate_fusion.py --dataset Dataset/INFRA-3DRC-Dataset                 # labelled boxes as detections
    python tools/evaluate_fusion.py --dataset Dataset/INFRA-3DRC-Dataset --model models/yolov8n.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import Calibration, FusionPipeline, scene_frames  # noqa: E402
from fusion.tracking import Tracker  # noqa: E402
from tools.detection_metrics import box_iou  # noqa: E402

NAMES = {1: "adult", 4: "bicycle", 5: "motorcycle", 6: "car", 7: "bus"}


def camera_labels(image):
    ann = json.loads((image.parent.parent / "camera_01__annotation" / (image.stem + ".json")).read_text())
    boxes, ids, names = [], [], []
    for a in ann["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.append([x, y, x + w, y + h])
        ids.append(a["track_id"])
        names.append(NAMES.get(a["category_id"], str(a["category_id"])))
    return np.array(boxes, float).reshape(-1, 4), np.array(ids), names


def radar_labels(scan):
    """Labelled radar points (N, 3) in the radar frame and their track ids."""
    ann = json.loads((scan.parent.parent / "radar_01__annotation" / (scan.stem + ".json")).read_text())
    pts, ids = [], []
    for obj in ann.get("objects", []):
        for p in obj["points"]:
            pts.append(p[6:9])
            ids.append(obj["track_id"])
    return np.array(pts, float).reshape(-1, 3), np.array(ids)


def point_ids(points, label_xyz, label_ids, tol=0.02):
    """Track id of each radar point (0 = unlabelled background), by nearest labelled point."""
    if len(label_xyz) == 0:
        return np.zeros(len(points), int)
    d = np.linalg.norm(points[:, None, :] - label_xyz[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return np.where(d[np.arange(len(points)), nearest] < tol, label_ids[nearest], 0)


def idf1(pairs):
    """IDF1 from (gt_id, predicted_id or None) per labelled box per frame."""
    gt_ids = sorted({g for g, _ in pairs})
    pr_ids = sorted({p for _, p in pairs if p is not None})
    if not pr_ids:
        return 0.0
    counts = np.zeros((len(gt_ids), len(pr_ids)))
    gi, pi = {g: i for i, g in enumerate(gt_ids)}, {p: i for i, p in enumerate(pr_ids)}
    for g, p in pairs:
        if p is not None:
            counts[gi[g], pi[p]] += 1
    r, c = linear_sum_assignment(-counts)
    idtp = counts[r, c].sum()
    n_gt, n_pred = len(pairs), sum(p is not None for _, p in pairs)
    return float(2 * idtp / (n_gt + n_pred))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--model", help="YOLO checkpoint; without it the labelled boxes are used as detections")
    p.add_argument("--scenes", nargs="*", help="scene folder names (default: all)")
    p.add_argument("--out", type=Path, default=Path("results/fusion_eval.json"))
    args = p.parse_args()

    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)

    scenes = sorted(args.dataset.glob("INFRA-3DRC_scene-*"))
    if args.scenes:
        scenes = [s for s in scenes if s.name in args.scenes]
    tally = dict(boxes=0, with_radar=0, fused=0, correct=0, wrong=0, missed=0)
    switches, id_pairs, frames_seen = 0, [], 0

    for scene in scenes:
        cal = Calibration.from_file(scene / "calibration.json")
        pipeline = FusionPipeline(model, cal)
        tracker = Tracker(cal.radar_to_ground)
        last_pred = {}
        for image, scan in scene_frames(scene):
            gt_box, gt_id, gt_names = camera_labels(image)
            detections = None if model else (gt_box, gt_names, np.ones(len(gt_box)))
            result = pipeline(image, scan, detections)
            tracker.update(result.objects, result.time)
            frames_seen += 1

            label_xyz, label_ids = radar_labels(scan)
            moving = [p for c in result.clusters + result.noise for p in c.points]
            moving_ids = point_ids(np.array(moving).reshape(-1, 3), label_xyz, label_ids)
            boxes = np.array([o.box for o in result.objects]).reshape(-1, 4)
            iou = box_iou(gt_box, boxes) if len(boxes) and len(gt_box) else np.zeros((len(gt_box), len(boxes)))
            for g in range(len(gt_box)):
                j = int(np.argmax(iou[g])) if iou.shape[1] else -1
                obj = result.objects[j] if j >= 0 and iou[g, j] >= 0.5 else None
                tally["boxes"] += 1
                has_radar = bool(np.any(moving_ids == gt_id[g]))
                tally["with_radar"] += has_radar
                if obj is not None and obj.radar:
                    tally["fused"] += 1
                    ids = point_ids(np.vstack([r.points for r in obj.radar]), label_xyz, label_ids)
                    majority = np.bincount(ids).argmax() if len(ids) else 0
                    tally["correct" if majority == gt_id[g] else "wrong"] += 1
                elif has_radar:
                    tally["missed"] += 1

                pred = obj.track_id if obj is not None else None
                id_pairs.append((f"{scene.name}/{gt_id[g]}", None if pred is None else f"{scene.name}/{pred}"))
                key = gt_id[g]
                if pred is not None:
                    if key in last_pred and last_pred[key] != pred:
                        switches += 1
                    last_pred[key] = pred
        print(f"{scene.name}: {tally['correct']} correct, {tally['wrong']} wrong, {tally['missed']} missed so far")

    report = {
        "detections": args.model or "labelled boxes",
        "scenes": len(scenes),
        "frames": frames_seen,
        "boxes": tally["boxes"],
        "boxes_with_moving_radar_returns": tally["with_radar"],
        "association_precision": tally["correct"] / max(tally["fused"], 1),
        "association_recall": tally["correct"] / max(tally["with_radar"], 1),
        "fused_boxes": tally["fused"],
        "wrong_associations": tally["wrong"],
        "missed_associations": tally["missed"],
        "id_switches": switches,
        "idf1": idf1(id_pairs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
