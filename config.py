"""Paths and tunable constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

INPUT_VIDEO_NAME = "video_short.mp4"
OUTPUT_SUFFIX = "_annotated.mp4"

POSE_MODEL_NAME = "yolov8l-pose.pt"
POSE_CONF = 0.4
FPS = 30
WINDOW_NAME = "High-Accuracy Factory Analysis"
SHOW_PREVIEW = True

ACTIVITY_THRESHOLD = 0.45
IDLE_FRAME_LIMIT = 30

KPT_L_WRIST, KPT_R_WRIST = 9, 10
KPT_L_SHOULDER, KPT_R_SHOULDER = 5, 6
KPT_L_HIP, KPT_R_HIP = 11, 12

COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

KPT_CONF_MIN = 0.2
IOU_MATCH_MIN = 0.1


def path_input_video():
    return VIDEOS_DIR / INPUT_VIDEO_NAME


def path_pose_model():
    return MODELS_DIR / POSE_MODEL_NAME


def path_output_video(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{input_path.stem}{OUTPUT_SUFFIX}"
