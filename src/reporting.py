"""Console summary and JSON export after a run."""
import json
from pathlib import Path

import config


def print_productivity_report(people_metrics, fps, input_video_path: Path):
    input_video_path = Path(input_video_path).resolve()

    print("\n" + "=" * 30)
    print("FINAL PRODUCTIVITY REPORT")
    print("=" * 30)
    for tid, m in people_metrics.items():
        active_sec = m["active_frames"] / fps
        idle_sec = m["idle_frames"] / fps
        zf = m.get("zone_frames", {})
        print(f"Worker {tid}:")
        print(f" - Active {active_sec:.1f}s | Idle {idle_sec:.1f}s | Move: {int(m['total_dist'])} px")
        z = {k: zf.get(k, 0) / fps for k in config.FLOOR_ZONE_KEYS}
        print(
            f" - Screen quadrants (s): TL {z['top_left']:.1f} | TR {z['top_right']:.1f} | "
            f"BL {z['bottom_left']:.1f} | BR {z['bottom_right']:.1f}"
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

    workers = []
    for tid, m in sorted(people_metrics.items()):
        active_sec = m["active_frames"] / fps
        idle_sec = m["idle_frames"] / fps
        zf = m.get("zone_frames", {k: 0 for k in config.FLOOR_ZONE_KEYS})
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
                    "zone_time_seconds": {
                        k: round(zf.get(k, 0) / fps, 2) for k in config.FLOOR_ZONE_KEYS
                    },
                    "zone_labels": dict(config.FLOOR_ZONE_LABELS),
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
        },
    }

    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nStats saved to: {stats_path}")
