import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from collections import defaultdict
import math

# -------------------------
# Load models
# -------------------------
yolo_model = YOLO("yolov8n.pt")  # lightweight model
tracker = DeepSort(max_age=30)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# -------------------------
# Metrics storage
# -------------------------
people_data = defaultdict(lambda: {
    "positions": [],
    "active_time": 0,
    "idle_time": 0,
    "last_movement": 0,
    "tasks": {
        "pick_place": 0,
        "lift_tray": 0
    }
})

# -------------------------
# Helper functions
# -------------------------
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def is_active(movement, threshold=5):
    return movement > threshold

# -------------------------
# Video input/output
# -------------------------
cap = cv2.VideoCapture("factory.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, 
                      (int(cap.get(3)), int(cap.get(4))))

frame_id = 0

# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # -------------------------
    # Detect people
    # -------------------------
    results = yolo_model(frame)[0]

    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        if int(cls) == 0:  # person class
            detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))

    # -------------------------
    # Track people
    # -------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        cx, cy = l + w//2, t + h//2

        # -------------------------
        # Pose estimation
        # -------------------------
        person_crop = frame[t:t+h, l:l+w]
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        left_hand = None
        right_hand = None

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            left_hand = (
                int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
            )

            right_hand = (
                int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
            )

            # Draw keypoints
            for lm in landmarks:
                px = int(lm.x * w) + l
                py = int(lm.y * h) + t
                cv2.circle(frame, (px, py), 2, (0,255,0), -1)

        # -------------------------
        # Movement tracking
        # -------------------------
        pdata = people_data[track_id]
        pdata["positions"].append((cx, cy))

        movement = 0
        if len(pdata["positions"]) > 1:
            movement = distance(pdata["positions"][-1], pdata["positions"][-2])

        # Active / Idle
        if is_active(movement):
            pdata["active_time"] += 1
        else:
            pdata["idle_time"] += 1

        # -------------------------
        # Task detection (heuristics)
        # -------------------------

        # Task 1: Pick & Place (hand moving quickly near bottom area)
        if left_hand and right_hand:
            hand_movement = distance(left_hand, right_hand)

            if hand_movement < 50 and cy > frame.shape[0]*0.6:
                pdata["tasks"]["pick_place"] += 1

        # Task 2: Lift tray (both hands high + vertical motion)
        if left_hand and right_hand:
            if left_hand[1] < h*0.5 and right_hand[1] < h*0.5:
                if movement > 8:
                    pdata["tasks"]["lift_tray"] += 1

        # -------------------------
        # Draw bounding box + info
        # -------------------------
        cv2.rectangle(frame, (l, t), (l+w, t+h), (255,0,0), 2)

        text = f"ID {track_id}"
        cv2.putText(frame, text, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    out.write(frame)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------
# Cleanup
# -------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

# -------------------------
# Print metrics
# -------------------------
print("\n=== METRICS ===")
for pid, data in people_data.items():
    print(f"\nPerson {pid}")
    print(f"Active time: {data['active_time']}")
    print(f"Idle time: {data['idle_time']}")
    print(f"Pick & Place: {data['tasks']['pick_place']}")
    print(f"Lift tray: {data['tasks']['lift_tray']}")