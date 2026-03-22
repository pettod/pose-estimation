"""Entry point: factory worker pose + tracking + activity overlay."""
from collections import defaultdict

import config
from src.inference import create_tracker, load_pose_model, select_inference_device
from src.metrics import new_person_metrics
from src.pipeline import run_pipeline
from src.reporting import print_productivity_report


def main():
    device = select_inference_device()
    print(f"Inference device: {device}")

    input_path = config.path_input_video()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_path}\n"
            f"Place .mp4 files under {config.VIDEOS_DIR}"
        )

    people_metrics = defaultdict(new_person_metrics)
    pose_model = load_pose_model(device)
    tracker = create_tracker()
    run_pipeline(input_path, pose_model, tracker, people_metrics, device)
    print_productivity_report(people_metrics, config.FPS)


if __name__ == "__main__":
    main()
