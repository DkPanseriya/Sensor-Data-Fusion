"""Explanatory figures for the README, computed from INFRA-3DRC.

    python tools/make_figures.py --dataset Dataset/INFRA-3DRC-Dataset
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import Calibration, FusionPipeline, scene_frames  # noqa: E402
from fusion.clustering import dbscan, filter_moving  # noqa: E402
from fusion.pcd import read_pcd  # noqa: E402
from fusion.tracking import H_POS, Tracker  # noqa: E402
from tools.evaluate_fusion import camera_labels, point_ids, radar_labels  # noqa: E402

OUT = Path("docs/figures")
COLOURS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _neighbours(xy, ranges, fixed=None):
    if fixed is None:
        r = (ranges[:, None] + ranges[None, :]) / 2
        a, b = 0.1 * np.sqrt(r) + 1.5, 0.25 * np.sqrt(r) + 1.5
    else:
        a = b = fixed
    dx, dy = xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]
    near = (dx / a) ** 2 + (dy / b) ** 2 < 1
    np.fill_diagonal(near, False)
    return near


def clustering_figure(scene, scan_name):
    """Same radar scan clustered with a small fixed, a large fixed and the range-adaptive neighbourhood."""
    scan = scene / "radar_01" / "radar_01__data" / scan_name
    cloud = filter_moving(read_pcd(scan))
    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], 1).astype(float)
    ranges = cloud["range"].astype(float)
    label_xyz, label_ids = radar_labels(scan)
    gt = point_ids(xyz, label_xyz, label_ids)

    variants = [("fixed 1.5 m", 1.5), ("fixed 4 m", 4.0), ("range-adaptive", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, (name, fixed) in zip(axes, variants):
        lab = dbscan(_neighbours(xyz[:, :2], ranges, fixed), 2)
        merged = sum(len(set(gt[lab == c]) - {0}) > 1 for c in set(lab) - {-1})
        split = sum(len(set(lab[gt == g]) - {-1}) > 1 for g in set(gt) - {0})
        verdict = ("objects merged" if merged else "") + (" and " if merged and split else "") + \
                  ("an object split" if split else "")
        for c in sorted(set(lab) - {-1}):
            pts = xyz[lab == c]
            ax.scatter(pts[:, 1], pts[:, 0], s=28, color=COLOURS[c % len(COLOURS)], zorder=3)
        noise = xyz[lab == -1]
        ax.scatter(noise[:, 1], noise[:, 0], s=22, marker="x", color="#adb5bd", zorder=2)
        for g in sorted(set(gt) - {0}):
            pts = xyz[gt == g]
            pad = 0.8
            ax.add_patch(plt.Rectangle((pts[:, 1].min() - pad, pts[:, 0].min() - pad),
                                       np.ptp(pts[:, 1]) + 2 * pad, np.ptp(pts[:, 0]) + 2 * pad,
                                       fill=False, ls="--", lw=1, color="#495057", zorder=1))
        if fixed is None:                                   # show how the neighbourhood grows with range
            for r in (ranges.min(), ranges.max()):
                i = int(np.argmin(np.abs(ranges - r)))
                a, b = 0.1 * np.sqrt(r) + 1.5, 0.25 * np.sqrt(r) + 1.5
                ax.add_patch(Ellipse((xyz[i, 1], xyz[i, 0]), 2 * b, 2 * a, fill=False, color="#868e96", lw=1))
        ax.set_title(f"{name}\n{verdict or 'one cluster per road user'}",
                     color="#c92a2a" if verdict else "#2b8a3e", fontsize=10)
        ax.set_xlabel("lateral, m")
        ax.invert_xaxis()
    axes[0].set_ylabel("forward, m")
    fig.suptitle("Radar returns get sparser with range: the neighbourhood has to grow with it", fontsize=11)
    fig.text(0.5, 0.005, "dashed: labelled road users   x: returns left as noise   grey ellipses: neighbourhood "
             "near and far", ha="center", fontsize=8.5, color="#495057")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "clustering_range_adaptive.png", dpi=150)
    plt.close(fig)


def camera_vs_radar_figure(dataset, step=3):
    """Camera ground position (bottom-centre through the homography) against the labelled radar position."""
    ranges, errors = [], []
    for scene in sorted(dataset.glob("INFRA-3DRC_scene-*")):
        cal = Calibration.from_file(scene / "calibration.json")
        for image, scan in scene_frames(scene)[::step]:
            boxes, ids, _ = camera_labels(image)
            label_xyz, label_ids = radar_labels(scan)
            if not len(boxes) or not len(label_xyz):
                continue
            bottom = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
            cam = cal.image_to_ground_xy(bottom)
            for k, tid in enumerate(ids):
                pts = label_xyz[label_ids == tid]
                if not len(pts):
                    continue
                nearest = pts[np.argmin(np.linalg.norm(pts, axis=1))]
                ref = cal.radar_to_ground_xy([[nearest[0], pts[:, 1].mean(), nearest[2]]])[0]
                ranges.append(np.linalg.norm(ref))
                errors.append(np.linalg.norm(cam[k] - ref))
    ranges, errors = np.array(ranges), np.array(errors)
    bins = np.arange(0, 80, 10)
    centre, med, lo, hi = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        e = errors[(ranges >= a) & (ranges < b)]
        if len(e) > 20:
            centre.append((a + b) / 2)
            med.append(np.median(e))
            lo.append(np.percentile(e, 25))
            hi.append(np.percentile(e, 75))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(ranges, errors, s=4, alpha=0.15, color="#1c7ed6")
    ax.fill_between(centre, lo, hi, color="#e8590c", alpha=0.2, label="middle 50%")
    ax.plot(centre, med, color="#e8590c", lw=2.5, marker="o", label="median")
    ax.set_xlabel("distance from the station (radar), m")
    ax.set_ylabel("camera error against radar, m")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 10)
    ax.set_title(f"A single camera's ground position drifts with distance ({len(errors):,} labelled boxes)",
                 fontsize=10.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "camera_vs_radar_position.png", dpi=150)
    plt.close(fig)


class PositionOnlyTracker(Tracker):
    """The same tracker without Doppler: radar objects update with their ground position only."""

    def _update(self, t, obj, position):
        r = self.r_assoc["radar"] if obj.radar else self.r_assoc["camera"]
        self._correct(t, position - t.position, H_POS, r * np.eye(2))

    def _new_track(self, obj, position, r):
        doppler, obj.radar = obj.radar, []
        t = super()._new_track(obj, position, r)
        obj.radar = doppler
        return t


def doppler_figure(dataset, seconds=3.0):
    """Speed estimate right after a track starts, with and without the Doppler measurement.
    Uses the fastest labelled road user that stays in view for at least `seconds`."""
    best = None
    for scene in sorted(dataset.glob("INFRA-3DRC_scene-*")):
        cal = Calibration.from_file(scene / "calibration.json")
        pipeline = FusionPipeline(None, cal)
        frames = []
        for image, scan in scene_frames(scene):
            boxes, ids, names = camera_labels(image)
            frames.append((pipeline(image, scan, (boxes, names, np.ones(len(boxes)))), ids))
        seen = {}
        for res, ids in frames:
            for o, gid in zip(res.objects, ids):
                if o.radar:
                    seen.setdefault(gid, []).append((res.time, abs(o.doppler)))
        for gid, s in seen.items():
            s = np.array(s)
            if s[-1, 0] - s[0, 0] >= seconds and np.median(s[:, 1]) > 3:
                score = np.median(s[:, 1])
                if best is None or score > best[0]:
                    best = (score, scene, gid, frames, cal)
    _, scene, gid, frames, cal = best

    curves = {}
    for name, tracker in (("plain", PositionOnlyTracker(cal.radar_to_ground)), ("ekf", Tracker(cal.radar_to_ground))):
        start, curve = None, []
        for res, ids in frames:
            objects = [o for o, g in zip(res.objects, ids) if g == gid and o.radar]
            if not objects:
                continue
            tracker.update(objects, res.time)
            start = res.time if start is None else start
            if res.time - start > seconds:
                break
            curve.append((res.time - start, np.linalg.norm(tracker.tracks[-1].velocity)))
        curves[name] = np.array(curve)
    doppler = np.array([(res.time, abs(o.doppler)) for res, ids in frames
                        for o, g in zip(res.objects, ids) if g == gid and o.radar])
    doppler[:, 0] -= doppler[0, 0]
    doppler = doppler[doppler[:, 0] <= seconds]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(doppler[:, 0], doppler[:, 1], s=14, color="#adb5bd", label="radar Doppler (radial speed)", zorder=1)
    ax.plot(*curves["plain"].T, color="#1c7ed6", lw=2.2, label="Kalman filter on position only")
    ax.plot(*curves["ekf"].T, color="#e8590c", lw=2.2, label="extended Kalman filter with Doppler")
    ax.set_xlabel("time since the track started, s")
    ax.set_ylabel("tracked speed, m/s")
    ax.set_ylim(0, None)
    ax.set_title("With Doppler the tracker knows the speed from the first frame", fontsize=10.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "doppler_tracking.png", dpi=150)
    plt.close(fig)
    print(f"doppler figure: {scene.name}, track {gid}")


def association_cases_figure():
    """Schematic of the ambiguous association cases and how each is resolved."""
    def box(ax, x, y, w, h, colour, label):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=2, color=colour))
        ax.add_patch(plt.Rectangle((x - 0.1 * w, y - 0.1 * h), 1.2 * w, 1.2 * h, fill=False, lw=1, ls="--",
                                   color=colour))
        ax.text(x + w / 2, y + h + 0.35, label, ha="center", fontsize=8.5, color=colour)

    def dot(ax, x, y, colour, marker="o", size=90):
        ax.scatter([x], [y], s=size, color=colour, marker=marker, edgecolor="black", lw=0.6, zorder=3)

    grey, a, b = "#adb5bd", "#e8590c", "#1c7ed6"
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    titles = ["one cluster, two boxes", "two clusters, one box", "crossing boxes", "box without a cluster"]
    notes = ["box nearer on the ground wins;\nthe other box waits for pass 2",
             "same speed and lateral position:\nmerged into one vehicle",
             "optimal assignment by\nground distance",
             "second pass with single\nunclustered returns"]
    ax = axes[0]
    box(ax, 1, 1, 3, 2.4, a, "car, 18 m")
    box(ax, 3.2, 1.6, 2.6, 2, grey, "car, 41 m")
    dot(ax, 3.6, 2.2, a)
    ax = axes[1]
    box(ax, 1, 1, 4, 2.4, a, "bus")
    dot(ax, 2, 2, a)
    dot(ax, 3.9, 2.3, a)
    ax.plot([2, 3.9], [2, 2.3], color=a, lw=1.5, ls=":")
    ax = axes[2]
    box(ax, 0.8, 1, 2.6, 2.3, a, "A")
    box(ax, 2.6, 1.3, 2.6, 2.3, b, "B")
    dot(ax, 2.9, 2.2, a)
    dot(ax, 4.6, 2.6, b)
    ax = axes[3]
    box(ax, 1.5, 1, 2.5, 2.4, a, "bicycle")
    dot(ax, 2.8, 1.6, a, marker="s", size=70)
    for x, y in [(4.8, 3.6), (0.6, 0.5)]:
        dot(ax, x, y, grey, marker="s", size=40)
    for ax, title, note in zip(axes, titles, notes):
        ax.set_xlim(0, 6.2)
        ax.set_ylim(-0.6, 4.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        ax.text(3.1, -0.5, note, ha="center", va="top", fontsize=8.5, color="#495057")
    fig.text(0.5, 0.98, "solid: camera box   dashed: 20% tolerance   dot: radar cluster   square: single radar return",
             ha="center", va="top", fontsize=8.5, color="#495057")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(OUT / "association_cases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clustering_figure(args.dataset / "INFRA-3DRC_scene-10", "radar_01__2023-07-06-18-41-46-432.pcd")
    camera_vs_radar_figure(args.dataset)
    association_cases_figure()
    doppler_figure(args.dataset)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
