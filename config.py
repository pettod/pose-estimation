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
SHOW_PREVIEW = False

# Active/idle (joint motion + hysteresis; scales pick/lift/move thresholds)
ACTIVITY_THRESHOLD = 0.4  # lower = need more motion to count as active
IDLE_STREAK_FRAMES = 30  # consecutive low-activity frames before showing IDLE
TRAIL_LENGTH_FRAMES = 30  # bbox-center path length (overlay)
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

# Re-ID: cosine similarity between DinoV2 embeddings (disappeared vs new track crop).
WORKER_SIMILARITY_THRESHOLD = 0.5

# Screen grid (time-in-cell): 8×16 cells; 16 bins along the wider frame side.
GRID_BINS_WIDE = 16
GRID_BINS_NARROW = 8
GRID_CELL_COUNT = GRID_BINS_WIDE * GRID_BINS_NARROW


def grid_dimensions_for_frame(frame_width: int, frame_height: int) -> tuple[int, int]:
    """
    Return (grid_rows, grid_cols). Landscape: 8 rows × 16 cols; portrait: 16 × 8.
    """
    fw = max(int(frame_width), 1)
    fh = max(int(frame_height), 1)
    if fw >= fh:
        return GRID_BINS_NARROW, GRID_BINS_WIDE
    return GRID_BINS_WIDE, GRID_BINS_NARROW


def grid_cell_index(
    cx: float, cy: float, frame_width: int, frame_height: int
) -> tuple[int, int, int]:
    """BBox center → (flat_index, row, col). Row grows down, col grows right."""
    fw = float(max(frame_width, 1))
    fh = float(max(frame_height, 1))
    rows, cols = grid_dimensions_for_frame(frame_width, frame_height)
    col = min(int((cx / fw) * cols), cols - 1)
    row = min(int((cy / fh) * rows), rows - 1)
    return row * cols + col, row, col


def grid_location_label(cx: float, cy: float, frame_width: int, frame_height: int) -> str:
    _, row, col = grid_cell_index(cx, cy, frame_width, frame_height)
    return f"r{row}c{col}"


def path_input_video():
    return VIDEOS_DIR / INPUT_VIDEO_NAME


def path_pose_model():
    return MODELS_DIR / POSE_MODEL_NAME


def path_output_video(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{input_path.stem}{OUTPUT_SUFFIX}"


def path_worker_id_images_dir(video_stem: str) -> Path:
    """Directory for first-frame bbox crops when a new track ID appears."""
    d = OUTPUT_DIR / "worker_ids" / video_stem
    d.mkdir(parents=True, exist_ok=True)
    return d
