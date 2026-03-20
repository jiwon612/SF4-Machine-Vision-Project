from core.utils import compute_iou
from core.fall import make_fall_state


def match_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    track_ids = list(tracks.keys())

    if not track_ids or not detections:
        return [], track_ids, list(range(len(detections)))

    used_tracks = set()
    used_dets   = set()
    candidates  = []

    for track_id in track_ids:
        tbox = tracks[track_id]["box"]
        for det_idx, dbox in enumerate(detections):
            iou = compute_iou(tbox, dbox)
            if iou >= iou_threshold:
                candidates.append((iou, track_id, det_idx))

    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_pairs = []
    for iou, track_id, det_idx in candidates:
        if track_id in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_id)
        used_dets.add(det_idx)
        matched_pairs.append((track_id, det_idx))

    unmatched_track_ids   = [tid for tid in track_ids        if tid not in used_tracks]
    unmatched_det_indices = [i   for i in range(len(detections)) if i   not in used_dets]
    return matched_pairs, unmatched_track_ids, unmatched_det_indices


def new_track_state(box, kpts, track_id):
    return {
        "box":    box,
        "kpts":   kpts,
        "name":   f"ID:{track_id}",
        "missed": 0,
        "ppe_state": {
            "helmet":      {"det_count": 0, "miss_count": 0, "stable": False},
            "gloves":      {"det_count": 0, "miss_count": 0, "stable": False},
            "safety_vest": {"det_count": 0, "miss_count": 0, "stable": False},
        },
        "fall_state":            make_fall_state(),
        "last_face_check_frame": -9999,
    }
