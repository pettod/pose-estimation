"""Per-person activity: joint motion + hysteresis ACTIVE/IDLE (YOLOv8 pose)."""
from collections import deque
from typing import Literal

import config

from . import geometry


def kpt_ok(kpts, kconf, idx):
    if kpts is None or kpts.shape[0] <= idx:
        return False
    if kconf is not None and kconf[idx] <= config.KPT_CONF_MIN:
        return False
    return kpts[idx][0] > 0 and kpts[idx][1] > 0


def new_person_metrics():
    return {
        "active_frames": 0,
        "idle_frames": 0,
        "total_dist": 0,
        "tasks": {"pick_place": 0, "lift_tray": 0, "move_rack": 0},
        "grid_cell_frames": [0] * config.GRID_CELL_COUNT,
        "last_pos": None,
        "prev_wrist_dist": None,
        "idle_streak": config.IDLE_STREAK_FRAMES,
        "position_trail": deque(maxlen=config.TRAIL_LENGTH_FRAMES),
    }


def update_movement_distance(metrics, curr_pos):
    last = metrics["last_pos"]
    move_dist = geometry.get_dist(curr_pos, last) if last else 0.0
    if last:
        metrics["total_dist"] += move_dist
    metrics["last_pos"] = curr_pos
    return move_dist


def record_grid_cell(metrics, cx, cy, frame_width, frame_height):
    idx, _, _ = config.grid_cell_index(cx, cy, frame_width, frame_height)
    metrics["grid_cell_frames"][idx] += 1


def record_position_trail(metrics, cx, cy):
    metrics["position_trail"].append((int(cx), int(cy)))


def _activity_kpt_ok(kpts, kconf, idx):
    if kpts is None or kpts.shape[0] <= idx:
        return False
    if kconf is not None and kconf[idx] <= config.ACTIVITY_KPT_CONF_MIN:
        return False
    return kpts[idx][0] > 0 and kpts[idx][1] > 0


def evaluate_activity(
    metrics,
    kpts,
    kconf,
    move_dist,
    frame_height,
    _cx,
    cy,
) -> Literal["ACTIVE", "IDLE"]:
    """
    Pick / lift / floor motion from wrist–torso signal + centroid move, with
    hysteresis: show IDLE only after IDLE_STREAK_FRAMES consecutive low-activity frames.

    active_frames / idle_frames match the **on-screen** pill (ACTIVE vs IDLE), not raw
    motion spikes: hysteresis “ACTIVE” frames count as active time so charts match the video.
    Task counters still increment only on high-motion frames.
    """
    th_scale = 0.5 / max(config.ACTIVITY_THRESHOLD, 0.05)
    wrist_activity = 0.0
    is_picking = False
    is_lifting = False
    is_moving_rack = move_dist > 5.0 * th_scale

    lw, rw = config.KPT_L_WRIST, config.KPT_R_WRIST
    lh, rh = config.KPT_L_HIP, config.KPT_R_HIP
    ls, rs = config.KPT_L_SHOULDER, config.KPT_R_SHOULDER

    def kpt_ok(idx):
        return _activity_kpt_ok(kpts, kconf, idx)

    if (
        kpts is not None
        and kpt_ok(rw)
        and kpt_ok(lw)
        and kpt_ok(lh)
        and kpt_ok(rh)
    ):
        r_wrist = kpts[rw]
        l_wrist = kpts[lw]
        torso_center = (kpts[lh] + kpts[rh]) / 2.0
        dist_r = geometry.get_dist(r_wrist[:2], torso_center[:2])
        prev = metrics["prev_wrist_dist"]
        if prev is not None:
            wrist_activity = abs(dist_r - prev)
        metrics["prev_wrist_dist"] = float(dist_r)

        is_picking = wrist_activity > 3.0 * th_scale and cy > frame_height * 0.4
        if kpt_ok(ls) and kpt_ok(rs):
            is_lifting = (
                r_wrist[1] < kpts[ls][1] or l_wrist[1] < kpts[rs][1]
            ) and wrist_activity > 5.0 * th_scale
    else:
        metrics["prev_wrist_dist"] = None

    frame_active = is_picking or is_lifting or is_moving_rack
    if frame_active:
        metrics["idle_streak"] = 0
        metrics["active_frames"] += 1
        if is_picking:
            metrics["tasks"]["pick_place"] += 1
        if is_lifting:
            metrics["tasks"]["lift_tray"] += 1
        if is_moving_rack:
            metrics["tasks"]["move_rack"] += 1
        return "ACTIVE"

    metrics["idle_streak"] += 1
    if metrics["idle_streak"] >= config.IDLE_STREAK_FRAMES:
        metrics["idle_frames"] += 1
        return "IDLE"

    metrics["active_frames"] += 1
    return "ACTIVE"


def evaluate_activity_and_update_metrics(
    metrics, kpts, kconf, frame_height, cx, cy, move_dist
):
    return evaluate_activity(
        metrics, kpts, kconf, move_dist, frame_height, cx, cy
    )
