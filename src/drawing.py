"""OpenCV overlays: grid lines, skeleton, per-person HUD."""
import cv2

import config

from . import geometry, metrics, pose


def draw_position_trail(frame, trail, track_color_bgr):
    """Filled dots only at last N bbox centers (no lines, no borders)."""
    if not trail:
        return
    b, g, r = track_color_bgr
    color = (min(255, b + 40), min(255, g + 40), min(255, r + 40))
    for pt in trail:
        cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)


def draw_grid_lines(frame, frame_width, frame_height):
    """8×16 grid (16 divisions on wider side); light lines."""
    w, h = frame_width, frame_height
    rows, cols = config.grid_dimensions_for_frame(frame_width, frame_height)
    color = (48, 48, 58)
    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(frame, (x, 0), (x, h), color, 1)
    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(frame, (0, y), (w, y), color, 1)


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
    location_label,
):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    status_bg = (0, 180, 0) if current_status == "ACTIVE" else (110, 110, 110)
    status_text_color = (255, 255, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    s1, t1 = 0.48, 1
    line1 = f"{current_status} | ID {track_id} | {int(total_dist)}px | {location_label}"

    (lw1, lh1), b1 = cv2.getTextSize(line1, font, s1, t1)
    pad = 4
    y1_base = max(22, y1 - 8)
    bg_x1 = max(0, x1 - pad)
    bg_y1 = max(0, y1_base - lh1 - pad)
    bg_x2 = x1 + lw1 + pad
    bg_y2 = y1_base + b1 + pad

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), status_bg, thickness=-1)
    cv2.putText(
        frame, line1, (x1, y1_base), font, s1, status_text_color, t1, cv2.LINE_AA
    )


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
        metrics.record_position_trail(m, cx, cy)
        move_dist = metrics.update_movement_distance(m, (cx, cy))
        metrics.record_grid_cell(m, cx, cy, frame_width, frame_height)
        current_status = metrics.evaluate_activity_and_update_metrics(
            m, kpts, kconf, frame_height, cx, cy, move_dist
        )
        location_label = config.grid_location_label(cx, cy, frame_width, frame_height)

        draw_pose_skeleton(frame, kpts, kconf, box_color)
        draw_position_trail(frame, list(m["position_trail"]), box_color)
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
            location_label,
        )
