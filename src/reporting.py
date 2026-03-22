"""Console summary and JSON export after a run."""
import json
from pathlib import Path

import cv2

import config


def _read_video_dimensions(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 1920, 1080
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return max(w, 1), max(h, 1)


def print_productivity_report(people_metrics, fps, input_video_path: Path):
    input_video_path = Path(input_video_path).resolve()

    print("\n" + "=" * 30)
    print("FINAL PRODUCTIVITY REPORT")
    print("=" * 30)
    for tid, m in people_metrics.items():
        active_sec = m["active_frames"] / fps
        idle_sec = m["idle_frames"] / fps
        print(f"Worker {tid}:")
        print(f" - Active {active_sec:.1f}s | Idle {idle_sec:.1f}s | Move: {int(m['total_dist'])} px")
        gcf = m.get("grid_cell_frames", [0] * config.GRID_CELL_COUNT)
        total_g = sum(gcf) or 1
        top = max(range(len(gcf)), key=lambda i: gcf[i]) if gcf else 0
        print(
            f" - Grid: {config.GRID_CELL_COUNT} cells; "
            f"top cell index {top} ({100.0 * gcf[top] / total_g:.0f}% of time)"
        )
        t = m["tasks"]
        print(
            f" - Tasks (frames): pick {t.get('pick_place', 0)} | "
            f"lift {t.get('lift_tray', 0)} | rack {t.get('move_rack', 0)}"
        )

    _write_stats_json(people_metrics, fps, input_video_path)


def _write_stats_json(people_metrics, fps, input_video_path: Path):
    input_video_path = Path(input_video_path).resolve()
    output_video_path = config.path_output_video(input_video_path).resolve()
    stats_path = config.STATS_JSON_PATH

    fw, fh = _read_video_dimensions(input_video_path)
    grid_rows, grid_cols = config.grid_dimensions_for_frame(fw, fh)

    workers = []
    for tid, m in sorted(people_metrics.items()):
        active_sec = m["active_frames"] / fps
        idle_sec = m["idle_frames"] / fps
        gcf = m.get("grid_cell_frames", [0] * config.GRID_CELL_COUNT)
        grid_sec = [round(c / fps, 4) for c in gcf]
        try:
            number = int(tid)
        except (TypeError, ValueError):
            number = tid
        t = m["tasks"]
        workers.append(
            {
                "number": number,
                "metrics": {
                    "active_seconds": round(active_sec, 2),
                    "idle_seconds": round(idle_sec, 2),
                    "total_movement_pixels": int(m["total_dist"]),
                    "grid_cell_seconds": grid_sec,
                },
                "active_frames": m["active_frames"],
                "idle_frames": m["idle_frames"],
                "tasks": {
                    "pick_place_small_object_frames": t.get("pick_place", 0),
                    "lift_tray_frames": t.get("lift_tray", 0),
                    "move_rack_frames": t.get("move_rack", 0),
                    "pick_place_approx": t.get("pick_place", 0) // 10,
                    "lift_tray_approx": t.get("lift_tray", 0) // 10,
                    "move_rack_approx": t.get("move_rack", 0) // 10,
                },
            }
        )

    payload = {
        "video_path": str(input_video_path),
        "output_video_path": str(output_video_path),
        "fps": fps,
        "statistics": {
            "worker_count": len(workers),
            "workers": workers,
            "grid": {
                "rows": grid_rows,
                "cols": grid_cols,
                "wide_axis": "width" if fw >= fh else "height",
                "cell_count": grid_rows * grid_cols,
                "frame_width": fw,
                "frame_height": fh,
            },
        },
    }

    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nStats saved to: {stats_path}")
