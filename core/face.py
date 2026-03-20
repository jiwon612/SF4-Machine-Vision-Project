import cv2
import numpy as np
import face_recognition

import config as cfg


def recognize_face_name(frame, person_box, known_encodings, known_names):
    x1, y1, x2, y2 = map(int, person_box)
    fh, fw = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(fw, x2)
    y2 = min(fh, y2)

    person_h = y2 - y1
    if person_h <= 0:
        return None

    extra_top  = int(person_h * cfg.FACE_TOP_EXPAND_RATIO)
    roi_top    = max(0, y1 - extra_top)
    roi_bottom = min(fh, y1 + int(person_h * cfg.UPPER_BODY_RATIO))

    if roi_bottom <= roi_top:
        return None

    roi = frame[roi_top:roi_bottom, x1:x2]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    if roi_w < cfg.MIN_ROI_W or roi_h < cfg.MIN_ROI_H:
        return None

    rgb_roi        = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_roi, model=cfg.FACE_MODEL)
    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)
    if not face_encodings:
        return None

    face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])
    best_idx = np.argmin(face_distances)

    if face_distances[best_idx] <= cfg.FACE_TOLERANCE:
        return known_names[best_idx]
    return None
