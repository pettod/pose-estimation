"""OpenCV overlays for skeleton and HUD."""
import cv2

import config

from . import geometry, metrics, pose


def draw_pose_skeleton(frame, kpts, kconf, box_color):
    if kpts is None:
        return
    for ki, kp in enumerate(kpts):
        kx, ky = int(kp[0]), int(kp[1])
        ok = (kconf is None) or (kconf[ki] > config.KPT_CONF_MIN)
        if ok and kx > 0 and ky > 0:
            cv2.circle(frame, (kx, ky), 4, box_color, -1)
    for a, b in config.COCO_SKELETON:
        if kpts.shape[0] <= max(a, b):
            continue
        if kconf is not None and (kconf[a] <= config.KPT_CONF_MIN or kconf[b] <= config.KPT_CONF_MIN):
            continue
        ax, ay = int(kpts[a][0]), int(kpts[a][1])
        bx, by = int(kpts[b][0]), int(kpts[b][1])
        if ax > 0 and ay > 0 and bx > 0 and by > 0:
            cv2.line(frame, (ax, ay), (bx, by), box_color, 2)


def draw_worker_labels(
    frame, x1, y1, x2, y2, track_id, box_color, current_status, total_dist
):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    status_bg = (0, 180, 0) if current_status == "ACTIVE" else (110, 110, 110)
    status_text_color = (255, 255, 255)

    font_top = cv2.FONT_HERSHEY_SIMPLEX
    scale_top = 0.5
    thick_top = 2
    y_top = y1 - 10
    rest_top = f" | Worker {track_id} | Dist: {int(total_dist)}px"

    (sw, sh), baseline = cv2.getTextSize(
        current_status, font_top, scale_top, thick_top
    )
    pad = 4
    bg_x1 = max(0, x1 - pad)
    bg_y1 = max(0, y_top - sh - pad)
    bg_x2 = x1 + sw + pad
    bg_y2 = y_top + baseline + pad
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), status_bg, thickness=-1)
    cv2.putText(
        frame, current_status, (x1, y_top), font_top, scale_top, status_text_color, thick_top
    )
    cv2.putText(
        frame, rest_top, (x1 + sw, y_top), font_top, scale_top, box_color, thick_top
    )


def process_tracks_on_frame(
    frame,
    frame_height,
    tracks,
    det_xyxy,
    frame_keypoints,
    frame_kpt_conf,
    people_metrics,
):
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        box_color = geometry.color_bgr_for_track(track_id)

        kpts, kconf = pose.match_keypoints_to_track_box(
            x1, y1, x2, y2, det_xyxy, frame_keypoints, frame_kpt_conf
        )

        m = people_metrics[track_id]
        move_dist = metrics.update_movement_distance(m, (cx, cy))
        current_status = metrics.evaluate_activity_and_update_metrics(
            m, kpts, kconf, frame_height, cx, cy, move_dist
        )

        draw_pose_skeleton(frame, kpts, kconf, box_color)
        draw_worker_labels(
            frame,
            x1,
            y1,
            x2,
            y2,
            track_id,
            box_color,
            current_status,
            m["total_dist"],
        )
