"""Video capture → pose → track → draw → encode."""
from pathlib import Path

import cv2
from tqdm import tqdm

import config

from . import drawing, pose
from .h264_transcode import transcode_mp4_to_h264


def run_pipeline(input_video, pose_model, tracker, people_metrics, device):
    input_path = Path(input_video)
    output_path = config.path_output_video(input_path)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, (width, height)
    )

    processed = 0
    pbar = tqdm(
        desc="Frames",
        unit="frame",
        total=total_frames if total_frames > 0 else None,
    )
    try:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                processed += 1

                results = pose_model.predict(
                    frame,
                    conf=config.POSE_CONF,
                    verbose=False,
                    device=device,
                    half=(device == "cuda"),
                )
                pose_result = results[0]
                detections, det_xyxy, frame_keypoints, frame_kpt_conf = (
                    pose.build_pose_detections(pose_result)
                )
                tracks = tracker.update_tracks(detections, frame=frame)

                drawing.draw_grid_lines(frame, width, height)
                drawing.process_tracks_on_frame(
                    frame,
                    width,
                    height,
                    tracks,
                    det_xyxy,
                    frame_keypoints,
                    frame_kpt_conf,
                    people_metrics,
                )

                out.write(frame)
                pbar.update(1)
                if config.SHOW_PREVIEW:
                    cv2.imshow(config.WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            pbar.close()
            if processed:
                print(f"Finished: {processed} frame(s) processed.")
    finally:
        cap.release()
        out.release()
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()
        if processed:
            transcode_mp4_to_h264(output_path)
