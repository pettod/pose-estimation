"""Console summary after a run."""


def print_productivity_report(people_metrics, fps):
    print("\n" + "=" * 30)
    print("FINAL PRODUCTIVITY REPORT")
    print("=" * 30)
    for tid, m in people_metrics.items():
        active_sec = m["active_frames"] / fps
        print(f"Worker {tid}:")
        print(f" - Movement: {int(m['total_dist'])} pixels")
        print(f" - Productivity Score: {active_sec:.1f}s active")
        print(
            f" - Tasks: pick_place ~{m['tasks']['pick_place'] // 10}, "
            f"move_rack ~{m['tasks']['move_rack'] // 10}"
        )
