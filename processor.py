import config as cfg
from core.ppe     import parse_ppe_results, detect_person_states, update_binary_state, person_bbox_from_keypoints
from core.face    import recognize_face_name
from core.fall    import compute_person_metrics, update_fall_state
from core.tracker import match_detections_to_tracks, new_track_state
from core.draw    import draw_ppe_boxes, draw_person_annotations
from core.logger  import add_log


def process_frame(frame, tracks, next_track_id, pose_model, ppe_model,
                  known_encodings, known_names, frame_idx=0, named_only=False):
    """
    Detect pose, PPE, faces and falls in one frame.

    named_only=True  → PPE/fall processing only for recognized persons (web server mode)
    named_only=False → process all tracked persons (standalone mode)
    """
    pose_results = pose_model.predict(frame, conf=cfg.POSE_CONF, device=cfg.YOLO_DEVICE, verbose=False)
    ppe_results  = ppe_model.predict(frame, conf=cfg.PPE_CONF,  device=cfg.YOLO_DEVICE, verbose=False)

    ppe_dets          = parse_ppe_results(ppe_results)
    person_detections = []

    if pose_results and pose_results[0].keypoints is not None:
        for person_kpts in pose_results[0].keypoints.xy.cpu().numpy():
            kpts = [tuple(pt) for pt in person_kpts]
            pb   = person_bbox_from_keypoints(kpts)
            if pb is not None:
                person_detections.append({"box": pb, "kpts": kpts})

    matched_pairs, unmatched_track_ids, unmatched_det_indices = match_detections_to_tracks(
        [d["box"] for d in person_detections], tracks, cfg.IOU_THRESHOLD
    )

    for track_id, det_idx in matched_pairs:
        tracks[track_id]["box"]    = person_detections[det_idx]["box"]
        tracks[track_id]["kpts"]   = person_detections[det_idx]["kpts"]
        tracks[track_id]["missed"] = 0

    for track_id in unmatched_track_ids:
        tracks[track_id]["missed"] += 1

    for tid in [tid for tid, info in tracks.items() if info["missed"] > cfg.MAX_MISSED]:
        del tracks[tid]

    for det_idx in unmatched_det_indices:
        det = person_detections[det_idx]
        tracks[next_track_id] = new_track_state(det["box"], det["kpts"], next_track_id)
        next_track_id += 1

    # Face recognition (interval-limited)
    for track_id, info in tracks.items():
        if not info["name"].startswith("ID:"):
            continue
        if frame_idx - info.get("last_face_check_frame", -9999) < cfg.FACE_RECOG_INTERVAL:
            continue
        info["last_face_check_frame"] = frame_idx
        matched_name = recognize_face_name(frame, info["box"], known_encodings, known_names)
        if matched_name is not None:
            info["name"] = matched_name

    annotated = frame.copy()

    if cfg.SHOW_PPE_BOXES:
        draw_ppe_boxes(annotated, ppe_dets)

    for track_id, info in tracks.items():
        name = info["name"]
        if named_only and name.startswith("ID:"):
            continue

        states = detect_person_states(info["kpts"], ppe_dets)
        if states is None:
            continue

        update_binary_state(states["raw_helmet"],      info["ppe_state"], "helmet",
                            cfg.HELMET_ON_THRESHOLD, cfg.HELMET_OFF_THRESHOLD)
        update_binary_state(states["raw_gloves"],      info["ppe_state"], "gloves",
                            cfg.GLOVE_ON_THRESHOLD,  cfg.GLOVE_OFF_THRESHOLD)
        update_binary_state(states["raw_safety_vest"], info["ppe_state"], "safety_vest",
                            cfg.VEST_ON_THRESHOLD,   cfg.VEST_OFF_THRESHOLD)

        helmet_stable = info["ppe_state"]["helmet"]["stable"]
        gloves_stable = info["ppe_state"]["gloves"]["stable"]
        vest_stable   = info["ppe_state"]["safety_vest"]["stable"]

        fallen = False
        if cfg.ENABLE_FALL_DETECTION:
            fallen = update_fall_state(info["fall_state"],
                                       compute_person_metrics(info["box"], info["kpts"]))

        draw_person_annotations(annotated, info["box"], info["kpts"], name, track_id,
                                states, helmet_stable, gloves_stable, vest_stable, fallen)

        if not name.startswith("ID:"):
            add_log(track_id, name, helmet_stable, gloves_stable, vest_stable, fallen)

    return annotated, tracks, next_track_id
