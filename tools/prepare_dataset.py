"""Build a YOLO training set from the INFRA-3DRC camera frames.

    python tools/prepare_dataset.py --dataset Dataset/INFRA-3DRC-Dataset --out data/yolo \
        --val-scenes INFRA-3DRC_scene-09 --upsample motorcycle=4 bus=1 bicycle=1
    python tools/prepare_dataset.py --dataset Dataset/INFRA-3DRC-Dataset --out data/yolo_original \
        --val-list tools/splits/original_val.txt          # the split the shipped checkpoints were trained on

Class balance: cars dominate the dataset (about 2,000 training boxes against 64 motorcycles). `--upsample`
adds augmented copies of training frames that contain a rare class. In those copies the over-represented
classes (`--suppress`, cars by default) are blurred out and their labels removed, so a rare class gains
examples without the majority class gaining them too. Copies of validation frames are never made.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# INFRA-3DRC category id -> YOLO class
CLASSES = {1: (0, "adult"), 4: (1, "bicycle"), 5: (2, "motorcycle"), 6: (3, "car"), 7: (4, "bus")}
NAME_TO_ID = {name: i for i, name in CLASSES.values()}


def annotations(annotation_file):
    """[(class id, (x, y, w, h) pixels)] and the image size."""
    ann = json.loads(Path(annotation_file).read_text())
    boxes = [(CLASSES[a["category_id"]][0], a["bbox"]) for a in ann["annotations"] if a["category_id"] in CLASSES]
    return boxes, ann["image"]["width"], ann["image"]["height"]


def yolo_lines(boxes, w, h):
    return [f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}" for c, (x, y, bw, bh) in boxes]


def blur_out(image, boxes, margin=0.15):
    """Hide objects by blurring their (slightly enlarged) boxes."""
    img = image.copy()
    for _, (x, y, bw, bh) in boxes:
        x0, y0 = int(max(x - margin * bw, 0)), int(max(y - margin * bh, 0))
        x1, y1 = int(min(x + bw * (1 + margin), img.shape[1])), int(min(y + bh * (1 + margin), img.shape[0]))
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), sigmaX=max(bw, bh) / 6)
    return img


PHOTOMETRIC = ("colour", "brightness", "contrast and noise")


def photometric(image, rng, kind):
    """Label-preserving changes of appearance."""
    img = image.astype(np.float32)
    if kind == "colour":
        img = img * rng.uniform(0.6, 1.4, size=3)
    elif kind == "brightness":
        img = img * rng.uniform(0.5, 1.4)
    else:
        img = (img - img.mean()) * rng.uniform(0.7, 1.3) + img.mean() + rng.normal(0, rng.uniform(5, 15), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path, help="folder holding the INFRA-3DRC_scene-* folders")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--val-scenes", nargs="*", help="hold out these whole scenes for validation (recommended)")
    p.add_argument("--val-fraction", type=float, default=0.2, help="frame-level split per scene when no scenes are given")
    p.add_argument("--val-list", type=Path, help="file with validation image names, one per line (overrides the above)")
    p.add_argument("--upsample", nargs="*", default=[], metavar="CLASS=N",
                   help="N augmented copies of every training frame that contains CLASS, e.g. motorcycle=4")
    p.add_argument("--suppress", nargs="*", default=["car"], help="classes blurred out and unlabelled in the copies")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; a fresh folder keeps an old split from leaking into the new one")
    copies = {NAME_TO_ID[k]: int(v) for k, v in (item.split("=") for item in args.upsample)}
    suppress = {NAME_TO_ID[name] for name in args.suppress}
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    val_names = set(args.val_list.read_text().split()) if args.val_list else None

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "copies": 0}
    for scene in sorted(args.dataset.glob("INFRA-3DRC_scene-*")):
        images = sorted((scene / "camera_01" / "camera_01__data").glob("*.png"))
        if val_names is not None:
            val = {i for i in images if i.name in val_names}
        elif args.val_scenes:
            val = set(images) if scene.name in args.val_scenes else set()
        else:
            val = set(rng.sample(images, round(len(images) * args.val_fraction)))

        for image in images:
            split = "val" if image in val else "train"
            boxes, w, h = annotations(scene / "camera_01" / "camera_01__annotation" / (image.stem + ".json"))
            shutil.copy(image, args.out / "images" / split / image.name)
            (args.out / "labels" / split / (image.stem + ".txt")).write_text("\n".join(yolo_lines(boxes, w, h)))
            counts[split] += 1

            n = max([copies.get(c, 0) for c, _ in boxes], default=0) if split == "train" else 0
            if n:
                hidden = [b for b in boxes if b[0] in suppress]
                kept = [b for b in boxes if b[0] not in suppress]
                base = blur_out(cv2.imread(str(image)), hidden)
                for k in range(n):
                    name = f"{image.stem}_up{k}"
                    cv2.imwrite(str(args.out / "images" / "train" / f"{name}.png"),
                                photometric(base, np_rng, PHOTOMETRIC[k % len(PHOTOMETRIC)]))
                    (args.out / "labels" / "train" / f"{name}.txt").write_text("\n".join(yolo_lines(kept, w, h)))
                    counts["copies"] += 1

    names = "\n".join(f"  {i}: {n}" for i, n in sorted(CLASSES.values()))
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"{counts['train']} training frames (+{counts['copies']} upsampled copies), "
          f"{counts['val']} validation frames -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
