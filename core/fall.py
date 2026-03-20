import time
import numpy as np
from collections import deque

import config as cfg
from core.utils import center_of_points, safe_mean, point_valid


def _angle_deg(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def _horiz_diff(angle):
    if angle is None:
        return None
    aa = angle % 180.0
    return float(min(abs(aa), abs(aa - 180.0)))


def _head_center_y(kpts):
    pts = [kpts[cfg.NOSE], kpts[cfg.LEFT_EYE], kpts[cfg.RIGHT_EYE],
           kpts[cfg.LEFT_EAR], kpts[cfg.RIGHT_EAR]]
    ys = [p[1] for p in pts if point_valid(p)]
    return float(np.mean(ys)) if ys else None


def compute_person_metrics(box, kpts):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    ls = kpts[cfg.LEFT_SHOULDER]
    rs = kpts[cfg.RIGHT_SHOULDER]
    lh = kpts[cfg.LEFT_HIP]
    rh = kpts[cfg.RIGHT_HIP]

    shoulder_center = center_of_points([ls, rs])
    hip_center      = center_of_points([lh, rh])
    hip_y           = hip_center[1] if hip_center is not None else None
    torso_angle     = _angle_deg(shoulder_center, hip_center)

    head_y              = _head_center_y(kpts)
    head_hip_gap_ratio  = None
    if head_y is not None and hip_y is not None and h > 1e-6:
        head_hip_gap_ratio = (hip_y - head_y) / h

    return {
        "bbox_h":              float(h),
        "bbox_w":              float(w),
        "aspect_wh":           float(w / h),
        "center_x":            float((x1 + x2) / 2.0),
        "center_y":            float((y1 + y2) / 2.0),
        "bottom_y":            float(y2),
        "head_y":              None if head_y is None else float(head_y),
        "hip_y":               None if hip_y   is None else float(hip_y),
        "torso_angle":         torso_angle,
        "head_hip_gap_ratio":  head_hip_gap_ratio,
        "torso_horizontal_diff": _horiz_diff(torso_angle),
    }


def make_fall_state():
    return {
        "history":         deque(maxlen=cfg.FALL_HISTORY_LEN),
        "candidate_count": 0,
        "recover_count":   0,
        "fallen":          False,
        "alert_until":     0.0,
    }


def update_fall_state(fall_state, metrics):
    now = time.time()
    fall_state["history"].append(metrics)

    hist = list(fall_state["history"])
    if len(hist) < cfg.FALL_BASELINE_LEN:
        return False

    baseline_hist  = hist[:-1]
    cur            = hist[-1]

    baseline_bbox_h  = safe_mean([h["bbox_h"]   for h in baseline_hist])
    baseline_center_y = safe_mean([h["center_y"] for h in baseline_hist])
    baseline_head_y  = safe_mean([h["head_y"]    for h in baseline_hist])
    baseline_aspect  = safe_mean([h["aspect_wh"] for h in baseline_hist])

    if baseline_bbox_h is None or baseline_bbox_h <= 1e-6:
        return fall_state["fallen"]

    height_ratio = cur["bbox_h"] / baseline_bbox_h
    center_drop  = 0.0 if baseline_center_y is None else (cur["center_y"] - baseline_center_y) / baseline_bbox_h
    head_drop    = 0.0 if baseline_head_y is None or cur["head_y"] is None else (cur["head_y"] - baseline_head_y) / baseline_bbox_h

    fall_like_score = sum([
        height_ratio <= cfg.FALL_HEIGHT_RATIO_THRESHOLD,
        cur["aspect_wh"] >= cfg.FALL_ASPECT_RATIO_THRESHOLD,
        center_drop >= cfg.FALL_CENTER_DROP_THRESHOLD or head_drop >= cfg.FALL_CENTER_DROP_THRESHOLD * 0.8,
        cur["head_hip_gap_ratio"] is not None and cur["head_hip_gap_ratio"] <= cfg.FALL_HEAD_HIP_GAP_RATIO_THRESHOLD,
        cur["torso_horizontal_diff"] is not None and cur["torso_horizontal_diff"] <= cfg.FALL_TORSO_HORIZONTAL_THRESHOLD,
    ])

    if not fall_state["fallen"]:
        if fall_like_score >= 3:
            fall_state["candidate_count"] += 1
            fall_state["recover_count"]    = 0
            if fall_state["candidate_count"] >= cfg.FALL_CONFIRM_FRAMES:
                fall_state["fallen"]     = True
                fall_state["alert_until"] = now + cfg.FALL_ALERT_HOLD_SEC
        else:
            fall_state["candidate_count"] = 0
    else:
        recover_cond = (
            height_ratio > 0.82 and
            (baseline_aspect is None or cur["aspect_wh"] < max(1.05, baseline_aspect * 1.15))
        )
        if recover_cond:
            fall_state["recover_count"] += 1
            if fall_state["recover_count"] >= cfg.FALL_RECOVER_FRAMES:
                fall_state["fallen"]          = False
                fall_state["candidate_count"] = 0
                fall_state["recover_count"]   = 0
        else:
            fall_state["recover_count"] = 0

    return fall_state["alert_until"] > now or fall_state["fallen"]
