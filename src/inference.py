"""Device selection, YOLO weights, SORT tracker."""
import torch
from ultralytics import YOLO

import config

from .sort_tracker import SortTrackerAdapter


def select_inference_device():
    """Prefer Apple MPS, then CUDA, else CPU."""
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pose_model(device):
    weights = config.path_pose_model()
    if not weights.is_file():
        raise FileNotFoundError(
            f"Pose weights not found: {weights}\n"
            f"Place .pt files under {config.MODELS_DIR}"
        )
    model = YOLO(str(weights))
    model.to(device)
    return model


def create_tracker():
    """SORT (Kalman bbox + IoU), no appearance model. Tuned like prior DeepSort age/init."""
    return SortTrackerAdapter(max_age=30, min_hits=3, iou_threshold=0.3)
