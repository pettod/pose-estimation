"""Paths and tunable constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
STATS_JSON_PATH = PROJECT_ROOT / "stats.json"

INPUT_VIDEO_NAME = "video_medium.mp4"
OUTPUT_SUFFIX = "_annotated.mp4"

POSE_MODEL_NAME = "yolov8l-pose.pt"
POSE_CONF = 0.4
FPS = 30
WINDOW_NAME = "High-Accuracy Factory Analysis"
SHOW_PREVIEW = True

# Active/idle (joint motion + hysteresis; scales pick/lift/move thresholds)
ACTIVITY_THRESHOLD = 0.4  # lower = need more motion to count as active
IDLE_STREAK_FRAMES = 30  # consecutive low-activity frames before showing IDLE
ACTIVITY_KPT_CONF_MIN = 0.25  # keypoint visibility for wrist/torso activity checks

KPT_L_WRIST, KPT_R_WRIST = 9, 10
KPT_L_SHOULDER, KPT_R_SHOULDER = 5, 6
KPT_L_HIP, KPT_R_HIP = 11, 12

COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

KPT_CONF_MIN = 0.2  # skeleton overlay; activity uses ACTIVITY_KPT_CONF_MIN
IOU_MATCH_MIN = 0.1

# Screen quadrants (time-in-area): bbox center in image space, not world position.
FLOOR_ZONE_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")
FLOOR_ZONE_LABELS = {
    "top_left": "Top left",
    "top_right": "Top right",
    "bottom_left": "Bottom left",
    "bottom_right": "Bottom right",
}


def floor_zone_key(cx: float, cy: float, frame_width: int, frame_height: int) -> str:
    """Map person center to a screen quadrant (2x2 split)."""
    w = max(int(frame_width), 1)
    h = max(int(frame_height), 1)
    xf = cx / w
    yf = cy / h
    left = xf < 0.5
    top = yf < 0.5
    if top and left:
        return "top_left"
    if top and not left:
        return "top_right"
    if not top and left:
        return "bottom_left"
    return "bottom_right"


def path_input_video():
    return VIDEOS_DIR / INPUT_VIDEO_NAME


def path_pose_model():
    return MODELS_DIR / POSE_MODEL_NAME


def path_output_video(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{input_path.stem}{OUTPUT_SUFFIX}"
