"""Render README demo GIFs: the busiest stretch of each scene, with fusion and tracking.

    python tools/make_gifs.py --dataset Dataset/INFRA-3DRC-Dataset --scenes 04 22 --width 440
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import Calibration, FusionPipeline, scene_frames  # noqa: E402
from fusion.tracking import Tracker  # noqa: E402
from fusion.visualize import render  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--scenes", nargs="+", required=True, help="scene numbers, e.g. 04 22")
    p.add_argument("--model", default="models/yolov8l.pt")
    p.add_argument("--frames", type=int, default=40, help="length of the GIF")
    p.add_argument("--width", type=int, default=440)
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    for number in args.scenes:
        scene = args.dataset / f"INFRA-3DRC_scene-{number}"
        cal = Calibration.from_file(scene / "calibration.json")
        pipeline, tracker = FusionPipeline(model, cal), Tracker(cal.radar_to_ground)
        panels, fused = [], []
        for image, scan in scene_frames(scene):
            result = pipeline(image, scan)
            tracker.update(result.objects, result.time)
            panel = cv2.cvtColor(render(result), cv2.COLOR_BGR2RGB)
            height = round(panel.shape[0] * args.width / panel.shape[1])
            panels.append(cv2.resize(panel, (args.width, height), interpolation=cv2.INTER_AREA))
            fused.append(sum(1 for o in result.objects if o.radar))
        n = min(args.frames, len(panels))
        window = np.convolve(fused, np.ones(n), "valid")            # busiest stretch of the scene
        start = int(np.argmax(window))
        frames = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT,
                                              dither=Image.Dither.NONE) for f in panels[start:start + n]]
        out = args.out / f"scene_{number}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
        print(f"{out}: {len(frames)} frames, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
