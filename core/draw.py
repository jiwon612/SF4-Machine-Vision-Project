import cv2
import config as cfg
from core.utils import point_valid


def draw_box(frame, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_point(frame, pt, color, radius=4):
    if point_valid(pt):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)


def draw_ppe_boxes(frame, ppe_dets):
    _COLOR = {
        "helmet":      (0, 255, 255),
        "glove":       (255, 0, 255),
        "safety_vest": (0, 255, 0),
    }
    for det in ppe_dets:
        color = _COLOR.get(det["name"], (255, 200, 0))
        draw_box(frame, det["box"], color, f'{det["name"]} {det["conf"]:.2f}', 2)


def draw_person_annotations(frame, box, kpts, name, track_id,
                             states, helmet_stable, gloves_stable, vest_stable, fallen):
    x1, y1, x2, y2 = map(int, box)

    if fallen:
        box_color = (0, 0, 255)
    elif not name.startswith("ID:"):
        box_color = (0, 255, 0)
    else:
        box_color = (0, 165, 255)

    if cfg.DRAW_NAME_BOX:
        label = f"{name} ({track_id})" if cfg.DRAW_TRACK_ID else name
        if fallen:
            label = f"{label} FALL"
        draw_box(frame, box, box_color, label)

    if cfg.DRAW_KEYPOINTS:
        for pt in kpts:
            draw_point(frame, pt, (0, 200, 255), radius=3)

    if cfg.DRAW_GUIDE:
        if states["head_center"] is not None:
            cx, cy = int(states["head_center"][0]), int(states["head_center"][1])
            cv2.circle(frame, (cx, cy), int(max(5, states["head_radius"])), (0, 255, 255), 1)

        for hand_key in ("left_hand", "right_hand"):
            if point_valid(states[hand_key]):
                hx, hy = int(states[hand_key][0]), int(states[hand_key][1])
                cv2.circle(frame, (hx, hy), int(max(5, states["glove_radius"])), (255, 0, 255), 1)

        if states["torso_box"] is not None:
            draw_box(frame, states["torso_box"], (0, 255, 0), "torso", 1)

    if cfg.SHOW_PPE_STATUS_TEXT:
        tx     = x1 + cfg.STATUS_LEFT_MARGIN
        base_y = y2 - cfg.STATUS_BOTTOM_MARGIN

        for text, stable, offset in [
            ("safety_vest" if vest_stable   else "no_safety_vest", vest_stable,   0),
            ("gloves"      if gloves_stable else "no_gloves",       gloves_stable, cfg.STATUS_LINE_GAP),
            ("helmet"      if helmet_stable else "no_helmet",       helmet_stable, cfg.STATUS_LINE_GAP * 2),
        ]:
            color = (0, 255, 0) if stable else (0, 0, 255)
            cv2.putText(frame, text, (tx, base_y - offset),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.STATUS_TEXT_FONT_SCALE,
                        color, cfg.STATUS_TEXT_THICKNESS, cv2.LINE_AA)

    if cfg.SHOW_FALL_TEXT and fallen:
        cv2.putText(frame, "FALL!", (x1 + 8, max(y1 + 25, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3, cv2.LINE_AA)
