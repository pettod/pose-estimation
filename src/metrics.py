"""Per-person activity metrics and idle/active logic."""
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
        "tasks": {"pick_place": 0, "move_rack": 0},
        "last_pos": None,
        "prev_wrist_dist": None,
        "idle_streak": config.IDLE_FRAME_LIMIT,
    }


def update_movement_distance(metrics, curr_pos):
    last = metrics["last_pos"]
    move_dist = geometry.get_dist(curr_pos, last) if last else 0.0
    if last:
        metrics["total_dist"] += move_dist
    metrics["last_pos"] = curr_pos
    return move_dist


def evaluate_activity_and_update_metrics(
    metrics, kpts, kconf, frame_height, cx, cy, move_dist
):
    th_scale = 0.5 / max(config.ACTIVITY_THRESHOLD, 0.05)
    wrist_activity = 0.0
    is_picking = False
    is_lifting = False
    is_moving_rack = move_dist > 5.0 * th_scale

    if (
        kpts is not None
        and kpt_ok(kpts, kconf, config.KPT_R_WRIST)
        and kpt_ok(kpts, kconf, config.KPT_L_WRIST)
        and kpt_ok(kpts, kconf, config.KPT_L_HIP)
        and kpt_ok(kpts, kconf, config.KPT_R_HIP)
    ):
        r_wrist = kpts[config.KPT_R_WRIST]
        l_wrist = kpts[config.KPT_L_WRIST]
        torso_center = (kpts[config.KPT_L_HIP] + kpts[config.KPT_R_HIP]) / 2.0
        dist_r = geometry.get_dist(r_wrist[:2], torso_center[:2])
        prev = metrics["prev_wrist_dist"]
        if prev is not None:
            wrist_activity = abs(dist_r - prev)
        metrics["prev_wrist_dist"] = dist_r

        is_picking = wrist_activity > 3.0 * th_scale and cy > frame_height * 0.4
        if kpt_ok(kpts, kconf, config.KPT_L_SHOULDER) and kpt_ok(
            kpts, kconf, config.KPT_R_SHOULDER
        ):
            is_lifting = (
                r_wrist[1] < kpts[config.KPT_L_SHOULDER][1]
                or l_wrist[1] < kpts[config.KPT_R_SHOULDER][1]
            ) and wrist_activity > 5.0 * th_scale
    else:
        metrics["prev_wrist_dist"] = None

    frame_active = is_picking or is_lifting or is_moving_rack
    if frame_active:
        metrics["idle_streak"] = 0
        metrics["active_frames"] += 1
        if is_picking:
            metrics["tasks"]["pick_place"] += 1
        if is_moving_rack:
            metrics["tasks"]["move_rack"] += 1
        return "ACTIVE"

    metrics["idle_streak"] += 1
    metrics["idle_frames"] += 1
    if metrics["idle_streak"] >= config.IDLE_FRAME_LIMIT:
        return "IDLE"
    return "ACTIVE"
