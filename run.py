"""
Standalone safety monitoring — reads webcam or video file and saves annotated output.
Edit DEFAULT_INPUT_SOURCE in config.py to switch input source.
"""
import cv2
import pickle
import time
import torch
from ultralytics import YOLO

import config as cfg
from processor import process_frame


def main():
    print(f"[INFO] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[INFO] source: {cfg.DEFAULT_INPUT_SOURCE}")

    pose_model = YOLO(cfg.POSE_MODEL_PATH)
    ppe_model  = YOLO(cfg.PPE_MODEL_PATH)

    with open(cfg.PKL_PATH, "rb") as f:
        face_db = pickle.load(f)

    cap = cv2.VideoCapture(cfg.DEFAULT_INPUT_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {cfg.DEFAULT_INPUT_SOURCE}")
        return

    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    is_video = isinstance(cfg.DEFAULT_INPUT_SOURCE, str)

    out = None
    if cfg.SAVE_OUTPUT:
        out = cv2.VideoWriter(cfg.OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"[INFO] Saving to: {cfg.OUTPUT_PATH}")

    tracks        = {}
    next_track_id = 1
    frame_count   = 0
    prev_time     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated, tracks, next_track_id = process_frame(
            frame, tracks, next_track_id,
            pose_model, ppe_model,
            face_db["encodings"], face_db["names"],
            frame_idx=frame_count,
            named_only=False,
        )

        now       = time.time()
        fps_live  = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        cv2.putText(annotated, f"FPS: {fps_live:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if out is not None:
            out.write(annotated)

        if cfg.SHOW_WINDOW:
            cv2.imshow(cfg.WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 == ord(" "):
                        break
                    if k2 in (ord("q"), 27):
                        cap.release()
                        if out:
                            out.release()
                        cv2.destroyAllWindows()
                        return

        if frame_count % 30 == 0:
            print(f"[INFO] {frame_count} frames | tracks={len(tracks)}")

        if is_video and not cfg.SHOW_WINDOW:
            time.sleep(max(0, 1.0 / fps))

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] Done")


if __name__ == "__main__":
    main()
