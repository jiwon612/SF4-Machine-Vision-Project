import torch

# ── Models ───────────────────────────────────────────────────
POSE_MODEL_PATH = "yolo26s-pose.pt"
PPE_MODEL_PATH  = "best_v04_03.pt"
PKL_PATH        = "trained_faces_0320.pkl"

POSE_CONF   = 0.5
PPE_CONF    = 0.5
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ── Face recognition ──────────────────────────────────────────
FACE_MODEL           = "hog"
FACE_TOLERANCE       = 0.75
UPPER_BODY_RATIO     = 0.4
FACE_TOP_EXPAND_RATIO = 0.20
MIN_ROI_W            = 60
MIN_ROI_H            = 60
FACE_RECOG_INTERVAL  = 8   # frames between recognition attempts; set 1 for every frame

# ── Person tracking ───────────────────────────────────────────
IOU_THRESHOLD = 0.30
MAX_MISSED    = 20

# ── PPE search zones ──────────────────────────────────────────
HAND_EXTENSION_RATIO = 0.35
GLOVE_RADIUS_RATIO   = 0.40
MIN_GLOVE_RADIUS     = 50

HEAD_EYE_RATIO  = 2.3
HEAD_EAR_RATIO  = 2.3
MIN_HEAD_RADIUS = 30

TORSO_PADDING_RATIO = 0.20
VEST_IOU_THRESHOLD  = 0.30

# ── PPE stabilization ─────────────────────────────────────────
HELMET_ON_THRESHOLD  = 2
HELMET_OFF_THRESHOLD = 100
GLOVE_ON_THRESHOLD   = 2
GLOVE_OFF_THRESHOLD  = 100
VEST_ON_THRESHOLD    = 2
VEST_OFF_THRESHOLD   = 50

# ── Fall detection ────────────────────────────────────────────
ENABLE_FALL_DETECTION             = True
FALL_ALERT_HOLD_SEC               = 5.0
FALL_HISTORY_LEN                  = 20
FALL_BASELINE_LEN                 = 15
FALL_HEIGHT_RATIO_THRESHOLD       = 0.72
FALL_ASPECT_RATIO_THRESHOLD       = 1.25
FALL_CENTER_DROP_THRESHOLD        = 0.10
FALL_HEAD_HIP_GAP_RATIO_THRESHOLD = 0.24
FALL_TORSO_HORIZONTAL_THRESHOLD   = 40.0
FALL_CONFIRM_FRAMES               = 4
FALL_RECOVER_FRAMES               = 12

# ── Display ───────────────────────────────────────────────────
DRAW_NAME_BOX       = True
DRAW_TRACK_ID       = True
DRAW_KEYPOINTS      = True
DRAW_GUIDE          = True
SHOW_PPE_BOXES      = True
SHOW_PPE_STATUS_TEXT = True
SHOW_FALL_TEXT      = True

STATUS_TEXT_FONT_SCALE = 0.50
STATUS_TEXT_THICKNESS  = 2
STATUS_LINE_GAP        = 20
STATUS_BOTTOM_MARGIN   = 10
STATUS_LEFT_MARGIN     = 8

# ── Web server ────────────────────────────────────────────────
STREAM_JPEG_QUALITY    = 70
STREAM_MAX_WIDTH       = 960
PROCESS_EVERY_N_FRAMES = 1
DEFAULT_WEB_SOURCE     = "1"

# ── Standalone mode ───────────────────────────────────────────
DEFAULT_INPUT_SOURCE = 1   # int for webcam index, str for video file path
SAVE_OUTPUT          = True
OUTPUT_PATH          = "output.mp4"
SHOW_WINDOW          = True
WINDOW_NAME          = "Safety Monitor"

# ── Class names ───────────────────────────────────────────────
CLASS_NAME_MAP = {
    0: "glove",
    1: "helmet",
    2: "safety_vest",
}

# ── COCO keypoint indices ─────────────────────────────────────
NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16
