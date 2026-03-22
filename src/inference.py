"""Device selection, YOLO weights, DeepSort tracker."""
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

import config


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
    return DeepSort(max_age=30, n_init=3, nms_max_overlap=0.5)
