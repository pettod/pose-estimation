"""YOLO pose results → DeepSort detections + keypoint IoU matching.

Keypoints: one .cpu().numpy() per tensor per frame (MPS-friendly; avoid per-detection sync).
"""
import numpy as np

import config

from . import geometry


def build_pose_detections(pose_result):
    """
    From one frame's YOLO pose Results, build DeepSort detections and parallel
    keypoint lists for IoU matching.
    """
    detections = []
    det_xyxy = []
    frame_keypoints = []
    frame_kpt_conf = []

    if pose_result.boxes is None or len(pose_result.boxes) == 0:
        return detections, det_xyxy, frame_keypoints, frame_kpt_conf

    # Single host copy per tensor per frame (not inside the per-person loop).
    xyxy = pose_result.boxes.xyxy.detach().cpu().numpy()
    conf = pose_result.boxes.conf.detach().cpu().numpy()
    cls = pose_result.boxes.cls.detach().cpu().numpy()
    if pose_result.keypoints is not None:
        kpts_all = pose_result.keypoints.xy.detach().cpu().numpy()
        kconf_all = (
            pose_result.keypoints.conf.detach().cpu().numpy()
            if pose_result.keypoints.conf is not None
            else None
        )
    else:
        kpts_all = None
        kconf_all = None

    for i in range(len(xyxy)):
        if int(round(cls[i])) != 0:
            continue
        x1, y1, x2, y2 = xyxy[i]
        det_xyxy.append((float(x1), float(y1), float(x2), float(y2)))
        detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf[i]), "person"))
        if kpts_all is not None:
            frame_keypoints.append(kpts_all[i])
            if kconf_all is not None:
                frame_kpt_conf.append(kconf_all[i])
            else:
                frame_kpt_conf.append(np.ones(17, dtype=np.float32))
        else:
            frame_keypoints.append(np.zeros((17, 2), dtype=np.float32))
            frame_kpt_conf.append(np.zeros(17, dtype=np.float32))

    return detections, det_xyxy, frame_keypoints, frame_kpt_conf


def match_keypoints_to_track_box(
    x1, y1, x2, y2, det_xyxy, frame_keypoints, frame_kpt_conf
):
    if not det_xyxy or not frame_keypoints:
        return None, None

    best_j = -1
    best_iou = 0.0
    tb = (float(x1), float(y1), float(x2), float(y2))
    for j, dbox in enumerate(det_xyxy):
        iou = geometry.bbox_iou_xyxy(tb, dbox)
        if iou > best_iou:
            best_iou = iou
            best_j = j

    if best_j < 0 or best_iou < config.IOU_MATCH_MIN:
        return None, None

    kconf = frame_kpt_conf[best_j] if best_j < len(frame_kpt_conf) else None
    return frame_keypoints[best_j], kconf
