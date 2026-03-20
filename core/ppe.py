import config as cfg
from core.utils import (
    distance, point_valid, midpoint, get_valid_points,
    center_of_points, bbox_center, compute_iou, expand_box,
)


def make_hand_point(elbow, wrist, ratio=0.25):
    if not point_valid(elbow) or not point_valid(wrist):
        return None
    ex, ey = elbow
    wx, wy = wrist
    return (float(wx + (wx - ex) * ratio), float(wy + (wy - ey) * ratio))


def make_head_center(kpts):
    nose = kpts[cfg.NOSE]
    le   = kpts[cfg.LEFT_EYE]
    re   = kpts[cfg.RIGHT_EYE]
    lear = kpts[cfg.LEFT_EAR]
    rear = kpts[cfg.RIGHT_EAR]

    if point_valid(nose):
        return nose
    if point_valid(le) and point_valid(re):
        return midpoint(le, re)
    return center_of_points([le, re, lear, rear])


def make_head_radius(kpts):
    le   = kpts[cfg.LEFT_EYE]
    re   = kpts[cfg.RIGHT_EYE]
    lear = kpts[cfg.LEFT_EAR]
    rear = kpts[cfg.RIGHT_EAR]

    candidates = []
    if point_valid(le) and point_valid(re):
        candidates.append(distance(le, re) * cfg.HEAD_EYE_RATIO)
    if point_valid(le) and point_valid(lear):
        candidates.append(distance(le, lear) * cfg.HEAD_EAR_RATIO)
    if point_valid(re) and point_valid(rear):
        candidates.append(distance(re, rear) * cfg.HEAD_EAR_RATIO)
    if point_valid(le) and point_valid(rear):
        candidates.append(distance(le, rear) * 1.8)
    if point_valid(re) and point_valid(lear):
        candidates.append(distance(re, lear) * 1.8)

    if not candidates:
        return cfg.MIN_HEAD_RADIUS
    return max(cfg.MIN_HEAD_RADIUS, max(candidates))


def make_torso_box(kpts):
    ls = kpts[cfg.LEFT_SHOULDER]
    rs = kpts[cfg.RIGHT_SHOULDER]
    lh = kpts[cfg.LEFT_HIP]
    rh = kpts[cfg.RIGHT_HIP]

    pts = get_valid_points([ls, rs, lh, rh])
    if len(pts) < 3:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    shoulder_width  = distance(ls, rs) if point_valid(ls) and point_valid(rs) else 0
    shoulder_center = center_of_points([ls, rs])
    hip_center      = center_of_points([lh, rh])
    torso_height    = distance(shoulder_center, hip_center) if shoulder_center and hip_center else 0

    scale = max(shoulder_width, torso_height, 50)
    return expand_box([x1, y1, x2, y2], scale * cfg.TORSO_PADDING_RATIO)


def person_bbox_from_keypoints(kpts):
    pts = get_valid_points(kpts)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_ppe_results(results):
    detections = []
    if not results:
        return detections

    r = results[0]
    if r.boxes is None:
        return detections

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss  = r.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        detections.append({
            "box":    box.tolist(),
            "conf":   float(conf),
            "cls_id": int(cls_id),
            "name":   cfg.CLASS_NAME_MAP.get(int(cls_id), str(cls_id)),
            "center": bbox_center(box.tolist()),
        })
    return detections


def update_binary_state(raw_detected, state_dict, key, on_th=3, off_th=5):
    if raw_detected:
        state_dict[key]["det_count"]  += 1
        state_dict[key]["miss_count"]  = 0
        if state_dict[key]["det_count"] >= on_th:
            state_dict[key]["stable"] = True
    else:
        state_dict[key]["miss_count"] += 1
        state_dict[key]["det_count"]   = 0
        if state_dict[key]["miss_count"] >= off_th:
            state_dict[key]["stable"] = False


def detect_person_states(kpts, ppe_dets):
    head_center = make_head_center(kpts)
    head_radius = make_head_radius(kpts)
    torso_box   = make_torso_box(kpts)

    left_elbow  = kpts[cfg.LEFT_ELBOW]
    right_elbow = kpts[cfg.RIGHT_ELBOW]
    left_wrist  = kpts[cfg.LEFT_WRIST]
    right_wrist = kpts[cfg.RIGHT_WRIST]

    left_hand  = make_hand_point(left_elbow,  left_wrist,  cfg.HAND_EXTENSION_RATIO)
    right_hand = make_hand_point(right_elbow, right_wrist, cfg.HAND_EXTENSION_RATIO)

    left_arm_len  = distance(left_elbow,  left_wrist)  if point_valid(left_elbow)  and point_valid(left_wrist)  else 0
    right_arm_len = distance(right_elbow, right_wrist) if point_valid(right_elbow) and point_valid(right_wrist) else 0
    glove_radius  = max(cfg.MIN_GLOVE_RADIUS, max(left_arm_len, right_arm_len) * cfg.GLOVE_RADIUS_RATIO)

    helmet_found      = False
    left_glove_found  = False
    right_glove_found = False
    vest_found        = False

    for det in ppe_dets:
        name   = det["name"]
        center = det["center"]
        box    = det["box"]

        if name == "helmet" and head_center is not None:
            if distance(center, head_center) <= head_radius:
                helmet_found = True

        elif name == "glove":
            if point_valid(left_hand)  and distance(center, left_hand)  <= glove_radius:
                left_glove_found = True
            if point_valid(right_hand) and distance(center, right_hand) <= glove_radius:
                right_glove_found = True

        elif name == "safety_vest" and torso_box is not None:
            if compute_iou(box, torso_box) >= cfg.VEST_IOU_THRESHOLD:
                vest_found = True

    return {
        "raw_helmet":     helmet_found,
        "raw_gloves":     left_glove_found or right_glove_found,
        "raw_safety_vest": vest_found,
        "head_center":    head_center,
        "head_radius":    head_radius,
        "torso_box":      torso_box,
        "left_hand":      left_hand,
        "right_hand":     right_hand,
        "glove_radius":   glove_radius,
    }
