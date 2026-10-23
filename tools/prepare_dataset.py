"""Build a YOLO training set from the INFRA-3DRC camera frames.

    python tools/prepare_dataset.py --dataset Dataset/INFRA-3DRC-Dataset --out data/yolo
    python tools/prepare_dataset.py --dataset Dataset/INFRA-3DRC-Dataset --out data/yolo_original \
        --val-list tools/splits/original_val.txt          # the split the shipped checkpoints were trained on
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


def yolo_lines(annotation_file):
    ann = json.loads(Path(annotation_file).read_text())
    w, h = ann["image"]["width"], ann["image"]["height"]
    lines = []
    for item in ann["annotations"]:
        if item["category_id"] not in CLASSES:
            continue
        x, y, bw, bh = item["bbox"]
        lines.append(f"{CLASSES[item['category_id']][0]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def augment(image, rng):
    """Photometric only, so labels stay valid: Gaussian noise, contrast and brightness."""
    img = image.astype(np.float32)
    img = (img - img.mean()) * rng.uniform(0.75, 1.25) + img.mean()
    img = img * rng.uniform(0.8, 1.2) + rng.normal(0, rng.uniform(5, 20), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path, help="folder holding the INFRA-3DRC_scene-* folders")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--val-scenes", nargs="*", help="hold out these whole scenes for validation (recommended)")
    p.add_argument("--val-fraction", type=float, default=0.2, help="frame-level split per scene when no scenes are given")
    p.add_argument("--val-list", type=Path, help="file with validation image names, one per line (overrides the above)")
    p.add_argument("--augment", type=int, default=0, help="extra augmented copies of each training image")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    val_names = set(args.val_list.read_text().split()) if args.val_list else None

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
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
            label = scene / "camera_01" / "camera_01__annotation" / (image.stem + ".json")
            lines = yolo_lines(label)
            shutil.copy(image, args.out / "images" / split / image.name)
            (args.out / "labels" / split / (image.stem + ".txt")).write_text("\n".join(lines))
            counts[split] += 1
            if split == "train":
                for k in range(args.augment):
                    name = f"{image.stem}_aug{k}"
                    cv2.imwrite(str(args.out / "images" / "train" / f"{name}.png"), augment(cv2.imread(str(image)), np_rng))
                    (args.out / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines))

    names = "\n".join(f"  {i}: {n}" for i, n in sorted(CLASSES.values()))
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"{counts['train']} training and {counts['val']} validation frames -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
