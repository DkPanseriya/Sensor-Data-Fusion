"""Render a fused frame: annotated camera image next to a bird's-eye view of the ground plane."""
import cv2
import numpy as np

PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
           (213, 94, 0), (204, 121, 167), (255, 255, 255)]           # RGB, colour-blind safe
GREY = (150, 150, 150)
RADAR = (200, 40, 40)


def _bgr(rgb):
    return rgb[::-1]


def _dashed_rect(img, p1, p2, colour, thickness=2, dash=12):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, thickness)


def _label(img, text, org, colour, placed, scale=1.0):
    """Draw a filled label above `org`, moved up until it overlaps no label drawn before."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(org[0]), int(org[1])
    rect = lambda yy: (x, yy - h - 8, x + w + 10, yy + base)  # noqa: E731
    while any(not (r[2] < a[0] or r[0] > a[2] or r[3] < a[1] or r[1] > a[3]) for a in placed for r in [rect(y)]):
        y -= h + 12
    y = max(y, h + 8)
    x1, y1, x2, y2 = rect(y)
    placed.append((x1, y1, x2, y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
    cv2.putText(img, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 2, cv2.LINE_AA)


def colour_of(i, obj):
    """Stable colour per track; objects without a confirmed track use their index, grey without radar."""
    if not obj.radar and obj.track_id is None:
        return _bgr(GREY)
    key = obj.track_id if obj.track_id is not None else i
    return _bgr(PALETTE[key % len(PALETTE)])


def draw_image(frame, box_scale=1.2):
    img = cv2.imread(str(frame.image))
    cal = frame.calibration
    returns = [p for obj in frame.clusters + frame.noise for p in obj.points]
    for u, v in cal.radar_to_image(returns).astype(int):
        cv2.drawMarker(img, (int(u), int(v)), _bgr(RADAR), cv2.MARKER_TRIANGLE_UP, 14, 2)

    labels = []
    for i, obj in enumerate(frame.objects):
        colour = colour_of(i, obj)
        x1, y1, x2, y2 = obj.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * box_scale, (y2 - y1) * box_scale
        _dashed_rect(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), colour)
        for r in obj.radar:
            u, v = cal.radar_to_image([r.centroid])[0].astype(int)
            if obj.source == "cluster":
                cv2.circle(img, (int(u), int(v)), 13, colour, -1)
            else:
                cv2.rectangle(img, (int(u) - 11, int(v) - 11), (int(u) + 11, int(v) + 11), colour, 4)
        text = obj.label if not obj.radar else f"{obj.label} {obj.range_m:.0f} m {obj.doppler:+.1f} m/s"
        if obj.track_id is not None:
            text = f"#{obj.track_id} {text}"
        labels.append((text, (int(cx - w / 2), int(cy - h / 2) - 4), colour))

    placed = []                                    # labels last, nearest object first, so nothing covers them
    for text, org, colour in sorted(labels, key=lambda t: -t[1][1]):
        _label(img, text, org, colour, placed)
    return img


def draw_birdseye(frame, height, width=420, x_range=(0, 100), y_range=(-25, 25)):
    """Ground plane, x forward (up), y left (left)."""
    img = np.full((height, width, 3), 250, np.uint8)
    margin = 40

    def px(ground):
        x, y = ground
        u = margin + (y_range[1] - y) / (y_range[1] - y_range[0]) * (width - 2 * margin)
        v = height - margin - (x - x_range[0]) / (x_range[1] - x_range[0]) * (height - 2 * margin)
        return int(u), int(v)

    for x in range(x_range[0], x_range[1] + 1, 20):
        cv2.line(img, px((x, y_range[0])), px((x, y_range[1])), (225, 225, 225), 1)
        cv2.putText(img, f"{x} m", (6, px((x, 0))[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.drawMarker(img, px((0, 0)), (60, 60, 60), cv2.MARKER_TRIANGLE_UP, 18, 2)

    cal = frame.calibration
    returns = [p for obj in frame.clusters + frame.noise for p in obj.points]
    for g in cal.radar_to_ground_xy(returns):
        cv2.circle(img, px(g), 3, _bgr(RADAR), -1)

    for i, obj in enumerate(frame.objects):
        colour = colour_of(i, obj)
        if len(obj.trail) > 1:                                              # track history
            cv2.polylines(img, [np.array([px(g) for g in obj.trail], np.int32)], False, colour, 2, cv2.LINE_AA)
        if obj.velocity is not None:                                        # 1 s ahead at the tracked velocity
            head = obj.trail[-1]
            cv2.arrowedLine(img, px(head), px(head + obj.velocity), colour, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(img, px(obj.ground), 9, colour, 2)                       # camera estimate
        for r in obj.radar:
            g = cal.radar_to_ground_xy([r.reference])[0]
            cv2.circle(img, px(g), 7, colour, -1)                           # radar reference point
    cv2.putText(img, "bird's-eye view", (margin, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "ring: camera  dot: radar  line: track", (margin, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 90, 90), 1, cv2.LINE_AA)
    return img


def render(frame, width=1400, box_scale=1.2):
    """Side-by-side panel, BGR uint8."""
    image = draw_image(frame, box_scale)
    bev_w = 420
    img_w = width - bev_w
    image = cv2.resize(image, (img_w, int(image.shape[0] * img_w / image.shape[1])), interpolation=cv2.INTER_AREA)
    bev = draw_birdseye(frame, image.shape[0], bev_w)
    return np.hstack([image, bev])
