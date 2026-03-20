"""
Flask web server — MJPEG streaming + REST API for safety monitoring.

Endpoints:
  GET  /raw_feed         - raw camera stream (MJPEG)
  GET  /video_feed       - annotated detection stream (MJPEG)
  GET  /api/logs         - detection log (JSON)
  POST /api/reset_logs   - clear log
  POST /api/reset_count  - alias for reset_logs
  GET  /api/current_source
  POST /api/set_source   - body: {"source": "0"|"1"|"filename.mp4"}

Query params for feed endpoints: ?source=&width=&quality=
"""
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import torch

import config as cfg
from processor   import process_frame
from core.logger import get_logs, reset_logs as _reset_logs

app = Flask(__name__)
CORS(app)

# ── Startup ───────────────────────────────────────────────────
device     = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO(cfg.POSE_MODEL_PATH)
ppe_model  = YOLO(cfg.PPE_MODEL_PATH)

with open(cfg.PKL_PATH, "rb") as f:
    face_db = pickle.load(f)
known_encodings = face_db["encodings"]
known_names     = face_db["names"]

current_source = cfg.DEFAULT_WEB_SOURCE
reset_token    = 0

print(f"[INFO] device={device}  source={current_source}")

# ── Helpers ───────────────────────────────────────────────────

def parse_source(value):
    value = str(value or current_source).strip()
    if value.lower() == "webcam":
        return 0
    return int(value) if value.isdigit() else value


def _resize(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _encode(frame, quality):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def _boundary(data):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"


def _error_frame(msg):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


# ── Stream generators ─────────────────────────────────────────

def _gen_raw(src, max_width, quality):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        data = _encode(_error_frame(f"Failed to open: {src}"), quality)
        yield _boundary(data)
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            data = _encode(_resize(frame, max_width), quality)
            if data:
                yield _boundary(data)
    finally:
        cap.release()


def _gen_detected(src, max_width, quality):
    global reset_token

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        data = _encode(_error_frame(f"Failed to open: {src}"), quality)
        yield _boundary(data)
        return

    tracks        = {}
    next_track_id = 1
    frame_idx     = 0
    last_annotated = None
    local_token   = reset_token

    try:
        while True:
            if local_token != reset_token:
                tracks, next_track_id, frame_idx, last_annotated = {}, 1, 0, None
                local_token = reset_token

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if cfg.PROCESS_EVERY_N_FRAMES <= 1 or frame_idx % cfg.PROCESS_EVERY_N_FRAMES == 1 or last_annotated is None:
                last_annotated, tracks, next_track_id = process_frame(
                    frame, tracks, next_track_id,
                    pose_model, ppe_model, known_encodings, known_names,
                    frame_idx=frame_idx, named_only=True,
                )
            else:
                last_annotated = frame.copy()

            data = _encode(_resize(last_annotated, max_width), quality)
            if data:
                yield _boundary(data)
    finally:
        cap.release()


# ── Routes ────────────────────────────────────────────────────

@app.route("/raw_feed")
def raw_feed():
    src = parse_source(request.args.get("source", current_source))
    w   = int(request.args.get("width",   cfg.STREAM_MAX_WIDTH))
    q   = int(request.args.get("quality", cfg.STREAM_JPEG_QUALITY))
    return Response(_gen_raw(src, w, q), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed")
def video_feed():
    src = parse_source(request.args.get("source", current_source))
    w   = int(request.args.get("width",   cfg.STREAM_MAX_WIDTH))
    q   = int(request.args.get("quality", cfg.STREAM_JPEG_QUALITY))
    return Response(_gen_detected(src, w, q), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/logs")
def api_logs():
    return jsonify(get_logs())


@app.route("/api/reset_logs",  methods=["POST"])
@app.route("/api/reset_count", methods=["POST"])
def api_reset():
    global reset_token
    _reset_logs()
    reset_token += 1
    return jsonify({"message": "reset ok"})


@app.route("/api/current_source")
def api_current_source():
    return jsonify({"current_source": current_source,
                    "parsed_source":  str(parse_source(current_source))})


@app.route("/api/set_source", methods=["POST"])
def api_set_source():
    global current_source
    data  = request.get_json(silent=True) or {}
    value = data.get("source")
    if value is None:
        return jsonify({"error": "source is required"}), 400
    current_source = str(value)
    return jsonify({"message": "source updated", "current_source": current_source})


@app.route("/")
def index():
    return jsonify({
        "message": "Safety Monitoring Server",
        "device":  device,
        "routes":  ["/raw_feed", "/video_feed", "/api/logs",
                    "/api/reset_logs", "/api/reset_count",
                    "/api/current_source", "/api/set_source"],
        "examples": {
            "raw webcam":           "http://localhost:5000/raw_feed?source=webcam",
            "detect webcam":        "http://localhost:5000/video_feed?source=webcam",
            "detect (lightweight)": "http://localhost:5000/video_feed?source=webcam&width=800&quality=55",
            "detect cam1":          "http://localhost:5000/video_feed?source=1",
            "detect video":         "http://localhost:5000/video_feed?source=3명.mp4",
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
