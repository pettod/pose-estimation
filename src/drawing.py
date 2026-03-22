"""OpenCV overlays for skeleton, screen quadrants, and per-person HUD."""
import cv2

import config

from . import geometry, metrics, pose


def draw_floor_zone_overlay(frame, frame_width, frame_height):
    """2x2 split: vertical + horizontal midlines; quadrant labels (screen space)."""
    w, h = frame_width, frame_height
    mx = w // 2
    my = h // 2
    color = (60, 60, 72)
    cv2.line(frame, (mx, 0), (mx, h), color, 1)
    cv2.line(frame, (0, my), (w, my), color, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    th = 1
    placements = [
        (w // 4, 18, "Top left"),
        (3 * w // 4, 18, "Top right"),
        (w // 4, h - 10, "Bottom left"),
        (3 * w // 4, h - 10, "Bottom right"),
    ]
    for cx, y, text in placements:
        (tw, _), _ = cv2.getTextSize(text, font, scale, th)
        cv2.putText(
            frame,
            text,
            (int(cx - tw // 2), y),
            font,
            scale,
            (120, 120, 135),
            th,
            cv2.LINE_AA,
        )


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
    frame,
    x1,
    y1,
    x2,
    y2,
    track_id,
    box_color,
    current_status,
    total_dist,
    zone_key,
    task_counts,
):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    status_bg = (0, 180, 0) if current_status == "ACTIVE" else (110, 110, 110)
    status_text_color = (255, 255, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    s1, t1 = 0.48, 1
    s2, t2 = 0.42, 1
    zlabel = config.FLOOR_ZONE_LABELS.get(zone_key, zone_key)
    line1 = f"{current_status} | ID {track_id} | {int(total_dist)}px | Screen: {zlabel}"
    pp = task_counts.get("pick_place", 0)
    lt = task_counts.get("lift_tray", 0)
    mr = task_counts.get("move_rack", 0)
    line2 = f"Tasks: pick {pp} | lift {lt} | rack {mr}"

    (lw1, lh1), b1 = cv2.getTextSize(line1, font, s1, t1)
    (lw2, lh2), b2 = cv2.getTextSize(line2, font, s2, t2)
    pad = 4
    gap = 2
    y2_base = max(28, y1 - 8)
    y1_base = y2_base - lh2 - gap - lh1
    text_w = max(lw1, lw2)
    bg_x1 = max(0, x1 - pad)
    bg_y1 = max(0, y1_base - lh1 - pad)
    bg_x2 = x1 + text_w + pad
    bg_y2 = y2_base + b2 + pad

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), status_bg, thickness=-1)
    cv2.putText(
        frame, line1, (x1, y1_base), font, s1, status_text_color, t1, cv2.LINE_AA
    )
    cv2.putText(frame, line2, (x1, y2_base), font, s2, box_color, t2, cv2.LINE_AA)


def process_tracks_on_frame(
    frame,
    frame_width,
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
        metrics.record_zone_frame(m, cx, cy, frame_width, frame_height)
        current_status = metrics.evaluate_activity_and_update_metrics(
            m, kpts, kconf, frame_height, cx, cy, move_dist
        )
        zone_key = config.floor_zone_key(cx, cy, frame_width, frame_height)

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
            zone_key,
            m["tasks"],
        )
