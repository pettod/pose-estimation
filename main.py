import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from collections import defaultdict
import math

# -------------------------
# 1. Setup & Config
# -------------------------
# Using yolov8m (medium) for better human vs. box discrimination
model = YOLO("yolov8m.pt") 
tracker = DeepSort(max_age=20, n_init=3, nms_max_overlap=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6)

# Metrics Storage
people_metrics = defaultdict(lambda: {
    "active_frames": 0,
    "idle_frames": 0,
    "total_dist": 0,
    "tasks": {"pick_place": 0, "move_rack": 0},
    "last_pos": None
})

# Define a "Work Zone" (Polygon) to ignore the background IDs 6, 19, etc.
# Coordinates should be adjusted based on your specific camera angle
WORK_ZONE = np.array([[100, 400], [1100, 400], [1200, 700], [50, 700]], np.int32)

def get_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# -------------------------
# 2. Main Processing
# -------------------------
cap = cv2.VideoCapture("factory.mp4")
width  = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter("analyzed_factory.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO Detection with high confidence threshold (0.6) to avoid IDing boxes
    results = model.predict(frame, conf=0.6, classes=[0]) # Class 0 is Person
    
    detections = []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        # Filter: Only accept if in the lower half of the screen (work area)
        if y2 > 300: 
            detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))

    # Update Tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed(): continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # --- Biomechanics (Pose) ---
        roi = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
        if roi.size > 0:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb_roi)
            
            if res.pose_landmarks:
                # Get Wrist and Hip for task inference
                lm = res.pose_landmarks.landmark
                # Task Inference: Pick & Place (Hands near waist + low torso movement)
                rwrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
                if 0.4 < rwrist_y < 0.8: # Relative to crop
                    people_metrics[track_id]["tasks"]["pick_place"] += 1

        # --- Movement Metrics ---
        curr_pos = (cx, cy)
        if people_metrics[track_id]["last_pos"]:
            move_dist = get_dist(curr_pos, people_metrics[track_id]["last_pos"])
            people_metrics[track_id]["total_dist"] += move_dist
            
            if move_dist > 2: # Activity threshold
                people_metrics[track_id]["active_frames"] += 1
            else:
                people_metrics[track_id]["idle_frames"] += 1
        
        people_metrics[track_id]["last_pos"] = curr_pos

        # --- Visuals ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Worker {track_id} | Dist: {int(people_metrics[track_id]['total_dist'])}px"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Factory Analysis", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

# -------------------------
# 3. Final Analytics Report
# -------------------------
print("\n" + "="*30)
print("FINAL PRODUCTIVITY REPORT")
print("="*30)
for tid, m in people_metrics.items():
    active_sec = m['active_frames'] / 20
    print(f"Worker {tid}:")
    print(f" - Movement: {int(m['total_dist'])} pixels")
    print(f" - Productivity Score: {active_sec:.1f}s active")
    print(f" - Tasks: {m['tasks']['pick_place'] // 10} items packed")