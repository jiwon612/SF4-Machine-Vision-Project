import math
import numpy as np


def distance(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def point_valid(pt):
    return pt is not None and len(pt) == 2 and pt[0] > 0 and pt[1] > 0


def midpoint(p1, p2):
    return ((float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0)


def get_valid_points(points):
    return [p for p in points if point_valid(p)]


def center_of_points(points):
    valid = get_valid_points(points)
    if not valid:
        return None
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    return (float(np.mean(xs)), float(np.mean(ys)))


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def expand_box(box, pad):
    x1, y1, x2, y2 = box
    return [x1 - pad, y1 - pad, x2 + pad, y2 + pad]


def safe_mean(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return [
        max(0, min(w - 1, int(x1))),
        max(0, min(h - 1, int(y1))),
        max(0, min(w - 1, int(x2))),
        max(0, min(h - 1, int(y2))),
    ]
