"""Per-person activity: biomechanical ACTIVE/IDLE + quadrant time (YOLOv8l-Pose)."""
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
        "zone_frames": {k: 0 for k in config.FLOOR_ZONE_KEYS},
        "last_pos": None,
        "prev_wrist_extension": None,
        "idle_streak": 0,
    }


def update_movement_distance(metrics, curr_pos):
    last = metrics["last_pos"]
    move_dist = geometry.get_dist(curr_pos, last) if last else 0.0
    if last:
        metrics["total_dist"] += move_dist
    metrics["last_pos"] = curr_pos
    return move_dist


def record_zone_frame(metrics, cx, cy, frame_width, frame_height):
    z = config.floor_zone_key(cx, cy, frame_width, frame_height)
    if z in metrics["zone_frames"]:
        metrics["zone_frames"][z] += 1


def _biomechanical_signals(kpts, kconf, metrics, move_dist):
    """
    Returns (raw_active, wrist_extension_change, lifting, global_move).
    - Global displacement: centroid motion > GLOBAL_DISPLACEMENT_PX.
    - Wrist–torso extension: mean distance (wrists 9,10 to hip midpoint 11,12);
      active if |Δextension| > WRIST_EXTENSION_DELTA_PX vs previous frame.
    - Lifting: either wrist above its shoulder (y smaller in image coords).
    """
    global_move = move_dist > config.GLOBAL_DISPLACEMENT_PX
    wrist_extension_change = False
    lifting = False

    if kpts is None:
        metrics["prev_wrist_extension"] = None
        return global_move, wrist_extension_change, lifting, global_move

    lw, rw = config.KPT_L_WRIST, config.KPT_R_WRIST
    lh, rh = config.KPT_L_HIP, config.KPT_R_HIP
    ls, rs = config.KPT_L_SHOULDER, config.KPT_R_SHOULDER

    if not (
        kpt_ok(kpts, kconf, lw)
        and kpt_ok(kpts, kconf, rw)
        and kpt_ok(kpts, kconf, lh)
        and kpt_ok(kpts, kconf, rh)
    ):
        metrics["prev_wrist_extension"] = None
        raw = global_move
        return raw, False, False, global_move

    torso = (kpts[lh][:2] + kpts[rh][:2]) / 2.0
    d_l = geometry.get_dist(kpts[lw][:2], torso)
    d_r = geometry.get_dist(kpts[rw][:2], torso)
    extension = (d_l + d_r) / 2.0
    prev = metrics["prev_wrist_extension"]
    metrics["prev_wrist_extension"] = float(extension)
    if prev is not None:
        wrist_extension_change = (
            abs(extension - prev) > config.WRIST_EXTENSION_DELTA_PX
        )

    if kpt_ok(kpts, kconf, ls) and kpt_ok(kpts, kconf, rs):
        lifting = (kpts[lw][1] < kpts[ls][1]) or (kpts[rw][1] < kpts[rs][1])

    raw = global_move or wrist_extension_change or lifting
    return raw, wrist_extension_change, lifting, global_move


def evaluate_activity(
    metrics,
    kpts,
    kconf,
    move_dist,
    frame_height,
) -> Literal["ACTIVE", "IDLE"]:
    """
    Biomechanical ACTIVE/IDLE with idle_streak (IDLE_STREAK_FRAMES) debouncing.
    Updates active_frames / idle_frames / tasks to match the returned label.
    """
    _ = frame_height
    raw, wrist_ch, lifting, global_move = _biomechanical_signals(
        kpts, kconf, metrics, move_dist
    )

    if raw:
        metrics["idle_streak"] = 0
        metrics["active_frames"] += 1
        if wrist_ch:
            metrics["tasks"]["pick_place"] += 1
        if lifting:
            metrics["tasks"]["lift_tray"] += 1
        if global_move:
            metrics["tasks"]["move_rack"] += 1
        return "ACTIVE"

    metrics["idle_streak"] += 1
    if metrics["idle_streak"] < config.IDLE_STREAK_FRAMES:
        metrics["active_frames"] += 1
        return "ACTIVE"

    metrics["idle_frames"] += 1
    return "IDLE"


def evaluate_activity_and_update_metrics(
    metrics, kpts, kconf, frame_height, cx, cy, move_dist
):
    """Backward-compatible name; cx/cy unused (centroid already in move_dist path)."""
    _ = cx, cy
    return evaluate_activity(metrics, kpts, kconf, move_dist, frame_height)
