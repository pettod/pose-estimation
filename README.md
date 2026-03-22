# Factory worker activity

Tracks people in factory-floor video and shows idle/active times, per-person heatmaps on the floor plan, total movement, and heuristic task-related frame counts. A **logical worker ID** (stable across tracker ID swaps) is recovered with **DinoV2** embeddings when possible.

## Installation

```bash
pip install -r requirements.txt
```

Put your input **`.mp4`** under `videos/` (name matches `INPUT_VIDEO_NAME` in `config.py`). Place pose weights (e.g. `yolov8l-pose.pt`) under `models/`. First run may download **DinoV2** weights from Hugging Face.

## Run

```bash
python main.py
```

Outputs go to `output/` and `stats.json` at the project root. Then:

```bash
python server.py
```

Open **http://127.0.0.1:5000** for the dashboard.

## Technical overview

- **OpenCV** reads and writes video, draws overlays (grid, skeleton, trails, boxes), and crops person regions for snapshots.
- **PyTorch** runs the pose model and DinoV2; device selection prefers MPS, then CUDA, then CPU
- **yolov8l-pose** detects people and 2D keypoints per frame
- **SORT** tracks short-lived worker IDs
- **DinoV2** assigns worker IDs based on image similarity when SORT failed due to occlusions etc.

## Activity / idle detection

- **Whole-body motion:** moving bounding box counts as activity
- **Arm extension vs. arm-at-side:** it measures how far the wrist is from the **torso center** (mid-hip) and, more importantly, **how much that distance changes** from frame to frame. Large change means the arm is opening and closing—reaching away from the body and back—not sitting idle with the wrist close to the trunk and barely moving.
- **Lift-like shape:** if a wrist is **above** the shoulder line *and* that wrist–torso distance is still changing a lot, it is treated as overhead / lifting motion
- **Pick-like shape:** the same reach-motion signal, combined with the person being **lower** in the frame, is treated as low, forward reach (e.g. toward the floor)

## Total movement

- For each worker, a **sum of distances in screen pixels** is calculated between center point of the bbox.

## Time in different areas of the frame

- The frame has been divided into an 8×16 bin grid
- Each worker ID stores the count on how many times it was present per bin
- Per worker, each cell’s **% of that worker’s time** (sums to 100%) is visualised in a heatmap
- Heatmap is overlaid on the video frame
