import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import math


input_video = "video_short.mp4"
POSE_CONF = 0.5

# Activity / idle (joint motion + hysteresis)
ACTIVITY_THRESHOLD = 0.5  # scales pick/lift/move thresholds; lower = need more motion
IDLE_FRAME_LIMIT = 30  # consecutive low-activity frames before showing IDLE

# COCO keypoints: 9=L_Wrist, 10=R_Wrist, 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
KPT_L_WRIST, KPT_R_WRIST = 9, 10
KPT_L_SHOULDER, KPT_R_SHOULDER = 5, 6
KPT_L_HIP, KPT_R_HIP = 11, 12

# COCO 17 keypoint skeleton (excluding face detail)
COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

# -------------------------
# 1. Setup & Config (Pose)
# -------------------------
pose_model = YOLO("yolov8m-pose.pt")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.5)

def _new_person_metrics():
    return {
        "active_frames": 0,
        "idle_frames": 0,
        "total_dist": 0,
        "tasks": {"pick_place": 0, "move_rack": 0},
        "last_pos": None,
        "prev_wrist_dist": None,
        "idle_streak": IDLE_FRAME_LIMIT,  # start IDLE until sustained activity
    }


people_metrics = defaultdict(_new_person_metrics)

def get_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def color_bgr_for_track(track_id):
    """Stable, distinct BGR color per DeepSort track id (OpenCV HSV 0–179)."""
    try:
        tid = int(track_id)
    except (TypeError, ValueError):
        tid = abs(hash(str(track_id)))
    hue = (tid * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

# -------------------------
# 2. Main Processing
# -------------------------
output_video = input_video.replace(".mp4", "_analyzed.mp4")
cap = cv2.VideoCapture(input_video)
width  = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = pose_model.predict(frame, conf=POSE_CONF, verbose=False)
    res = results[0]

    detections = []
    det_xyxy = []
    frame_keypoints = []
    frame_kpt_conf = []

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        kpts_all = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        kconf_all = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints is not None and res.keypoints.conf is not None
            else None
        )

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

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        box_color = color_bgr_for_track(track_id)

        # Match this track's box to the closest pose detection (IoU)
        kpts = None
        kconf = None
        if det_xyxy and frame_keypoints:
            best_j = -1
            best_iou = 0.0
            tb = (float(x1), float(y1), float(x2), float(y2))
            for j, dbox in enumerate(det_xyxy):
                iou = bbox_iou_xyxy(tb, dbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.1:
                kpts = frame_keypoints[best_j]
                kconf = frame_kpt_conf[best_j] if best_j < len(frame_kpt_conf) else None

        curr_pos = (cx, cy)
        last = people_metrics[track_id]["last_pos"]
        move_dist = get_dist(curr_pos, last) if last else 0.0
        if last:
            people_metrics[track_id]["total_dist"] += move_dist
        people_metrics[track_id]["last_pos"] = curr_pos

        th_scale = 0.5 / max(ACTIVITY_THRESHOLD, 0.05)
        wrist_activity = 0.0
        is_picking = False
        is_lifting = False
        is_moving_rack = move_dist > 5.0 * th_scale

        def kpt_ok(idx):
            if kpts is None or kpts.shape[0] <= idx:
                return False
            if kconf is not None and kconf[idx] <= 0.25:
                return False
            return kpts[idx][0] > 0 and kpts[idx][1] > 0

        if (
            kpts is not None
            and kpt_ok(KPT_R_WRIST)
            and kpt_ok(KPT_L_WRIST)
            and kpt_ok(KPT_L_HIP)
            and kpt_ok(KPT_R_HIP)
        ):
            r_wrist = kpts[KPT_R_WRIST]
            l_wrist = kpts[KPT_L_WRIST]
            torso_center = (kpts[KPT_L_HIP] + kpts[KPT_R_HIP]) / 2.0
            dist_r = get_dist(r_wrist[:2], torso_center[:2])
            prev = people_metrics[track_id]["prev_wrist_dist"]
            if prev is not None:
                wrist_activity = abs(dist_r - prev)
            people_metrics[track_id]["prev_wrist_dist"] = dist_r

            is_picking = wrist_activity > 3.0 * th_scale and cy > height * 0.4
            if kpt_ok(KPT_L_SHOULDER) and kpt_ok(KPT_R_SHOULDER):
                is_lifting = (
                    r_wrist[1] < kpts[KPT_L_SHOULDER][1]
                    or l_wrist[1] < kpts[KPT_R_SHOULDER][1]
                ) and wrist_activity > 5.0 * th_scale
        else:
            people_metrics[track_id]["prev_wrist_dist"] = None

        frame_active = is_picking or is_lifting or is_moving_rack
        if frame_active:
            people_metrics[track_id]["idle_streak"] = 0
            people_metrics[track_id]["active_frames"] += 1
            current_status = "ACTIVE"
            if is_picking:
                people_metrics[track_id]["tasks"]["pick_place"] += 1
            if is_moving_rack:
                people_metrics[track_id]["tasks"]["move_rack"] += 1
        else:
            people_metrics[track_id]["idle_streak"] += 1
            people_metrics[track_id]["idle_frames"] += 1
            if people_metrics[track_id]["idle_streak"] >= IDLE_FRAME_LIMIT:
                current_status = "IDLE"
            else:
                current_status = "ACTIVE"

        # Status pill: green bg when ACTIVE, gray when IDLE; label always white (BGR)
        status_bg = (0, 180, 0) if current_status == "ACTIVE" else (110, 110, 110)
        status_text_color = (255, 255, 255)

        # Draw keypoints & skeleton (full frame coords)
        if kpts is not None:
            for ki, kp in enumerate(kpts):
                kx, ky = int(kp[0]), int(kp[1])
                ok = (kconf is None) or (kconf[ki] > 0.25)
                if ok and kx > 0 and ky > 0:
                    cv2.circle(frame, (kx, ky), 4, box_color, -1)
            for a, b in COCO_SKELETON:
                if kpts.shape[0] <= max(a, b):
                    continue
                if kconf is not None and (kconf[a] <= 0.25 or kconf[b] <= 0.25):
                    continue
                ax, ay = int(kpts[a][0]), int(kpts[a][1])
                bx, by = int(kpts[b][0]), int(kpts[b][1])
                if ax > 0 and ay > 0 and bx > 0 and by > 0:
                    cv2.line(frame, (ax, ay), (bx, by), box_color, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        font_top = cv2.FONT_HERSHEY_SIMPLEX
        scale_top = 0.5
        thick_top = 2
        y_top = y1 - 10
        rest_top = (
            f" | Worker {track_id} | "
            f"Dist: {int(people_metrics[track_id]['total_dist'])}px"
        )
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

    cv2.imshow("High-Accuracy Factory Analysis", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# -------------------------
# 3. Final Analytics Report
# -------------------------
print("\n" + "=" * 30)
print("FINAL PRODUCTIVITY REPORT")
print("=" * 30)
for tid, m in people_metrics.items():
    active_sec = m["active_frames"] / 20
    print(f"Worker {tid}:")
    print(f" - Movement: {int(m['total_dist'])} pixels")
    print(f" - Productivity Score: {active_sec:.1f}s active")
    print(f" - Tasks: pick_place ~{m['tasks']['pick_place'] // 10}, move_rack ~{m['tasks']['move_rack'] // 10}")
