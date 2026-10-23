"""Run camera-radar fusion over one INFRA-3DRC scene.

    python run_fusion.py --scene Dataset/INFRA-3DRC-Dataset/INFRA-3DRC_scene-15 --save fusion.mp4
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from fusion import Calibration, FusionPipeline, scene_frames
from fusion.tracking import Tracker
from fusion.visualize import render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, type=Path, help="scene folder with calibration.json, camera_01/ and radar_01/")
    p.add_argument("--model", default="models/yolov8l.pt", help="YOLOv8 detector checkpoint")
    p.add_argument("--conf", type=float, default=0.5, help="detector confidence threshold")
    p.add_argument("--save", type=Path, help="write the rendered frames to .mp4, .gif or a folder of .jpg")
    p.add_argument("--show", action="store_true", help="show frames in a window (q to quit)")
    p.add_argument("--max-frames", type=int, help="stop after this many frames")
    p.add_argument("--eps", type=float, nargs=2, default=(0.1, 0.25), metavar=("EPS_X", "EPS_Y"),
                   help="clustering ellipse growth with sqrt(range), forward and lateral")
    p.add_argument("--no-tracking", action="store_true", help="fuse each frame on its own, without tracks")
    p.add_argument("--min-neighbours", type=int, default=2, help="neighbours a radar return needs to seed a cluster")
    args = p.parse_args()
    if args.save and args.save.suffix not in ("", ".mp4", ".gif"):
        p.error("--save takes a .mp4 or .gif file, or a folder")

    pipeline = FusionPipeline(YOLO(args.model), Calibration.from_file(args.scene / "calibration.json"),
                              conf=args.conf, eps=tuple(args.eps), min_neighbours=args.min_neighbours)
    frames = scene_frames(args.scene)[: args.max_frames]
    tracker = None if args.no_tracking else Tracker(pipeline.calibration.radar_to_ground)

    writer, gif_frames = None, []
    if args.save and args.save.suffix == "":
        args.save.mkdir(parents=True, exist_ok=True)

    for n, (image, scan) in enumerate(frames, 1):
        result = pipeline(image, scan)
        if tracker:
            tracker.update(result.objects, result.time)
        panel = render(result)
        fused = sum(1 for o in result.objects if o.radar)
        print(f"[{n}/{len(frames)}] {image.name}: {len(result.objects)} detections, {fused} with radar, "
              f"{len(result.clusters)} clusters"
              + (f", {len(tracker.tracks)} tracks" if tracker else ""))

        if args.save:
            if args.save.suffix == ".mp4":
                if writer is None:
                    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                             (panel.shape[1], panel.shape[0]))
                writer.write(panel)
            elif args.save.suffix == ".gif":
                gif_frames.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite(str(args.save / f"{image.stem}.jpg"), panel)
        if args.show:
            cv2.imshow("camera-radar fusion", panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    if gif_frames:
        from PIL import Image
        frames_pil = [Image.fromarray(f) for f in gif_frames]
        frames_pil[0].save(args.save, save_all=True, append_images=frames_pil[1:], duration=100, loop=0)
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
