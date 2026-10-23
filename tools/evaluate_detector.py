"""Evaluate one or more YOLO checkpoints on a YOLO-format validation set.

    python tools/evaluate_detector.py --data data/yolo_original --models models/yolov8n.pt models/yolov8s.pt models/yolov8l.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.detection_metrics import IOU_THRESHOLDS, evaluate  # noqa: E402


def load_ground_truth(images, labels_dir):
    gt = {}
    for image in images:
        h, w = _image_size(image)
        label = labels_dir / (image.stem + ".txt")
        rows = np.loadtxt(label, ndmin=2) if label.exists() and label.stat().st_size else np.zeros((0, 5))
        cls = rows[:, 0].astype(int)
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        gt[image.name] = (cls, np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1))
    return gt


def _image_size(path):
    import cv2
    h, w = cv2.imread(str(path)).shape[:2]
    return h, w


def predict(model, images, imgsz):
    preds = {}
    for image in images:
        boxes = model.predict(str(image), imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0].boxes
        preds[image.name] = (boxes.cls.cpu().numpy().astype(int), boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())
    return preds


def plot(report, path):
    """Grouped bars: AP50-95 per class for each checkpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(report["models"])
    classes = list(report["models"][models[0]]["AP50_95_per_class"])
    colours = ["#a5d8ff", "#4dabf7", "#1864ab"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        values = [report["models"][m]["AP50_95_per_class"][c] or 0 for c in classes]
        ax.bar(np.arange(len(classes)) + (k - (len(models) - 1) / 2) * width, values, width,
               color=colours[k % len(colours)], label=f"{m}  (mAP50-95 {report['models'][m]['mAP50_95']:.2f})")
    gt = report["models"][models[0]]["ground_truth_per_class"]
    ax.set_xticks(range(len(classes)), [f"{c}\n({gt[c]})" for c in classes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP, IoU 0.50-0.95")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Detector checkpoints on the validation split ({report['images']} frames)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="dataset folder with images/val and labels/val")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, default=Path("results/detector_eval.json"))
    args = p.parse_args()

    images = sorted((args.data / "images" / "val").glob("*.png"))
    gt = load_ground_truth(images, args.data / "labels" / "val")
    report = {"images": len(images), "iou_thresholds": IOU_THRESHOLDS.tolist(), "models": {}}

    for path in args.models:
        model = YOLO(path)
        names = [model.names[i] for i in sorted(model.names)]
        ap = evaluate(predict(model, images, args.imgsz), gt, len(names))
        report["models"][Path(path).stem] = {
            "mAP50": float(np.nanmean(ap[:, 0])),
            "mAP50_95": float(np.nanmean(ap)),
            "AP50_95_per_class": {n: (None if np.isnan(ap[i, 0]) else float(np.mean(ap[i]))) for i, n in enumerate(names)},
            "ground_truth_per_class": {n: int(sum((c == i).sum() for c, _ in gt.values())) for i, n in enumerate(names)},
        }
        m = report["models"][Path(path).stem]
        print(f"{Path(path).stem:10s} mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50_95']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    plot(report, args.out.with_suffix(".png"))
    print(f"-> {args.out}, {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
