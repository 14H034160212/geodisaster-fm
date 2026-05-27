"""File watcher: rebuild the static report whenever experiment outputs change.

Polls a small set of result files (CSVs, comparison JSON, reproducibility
manifest, figures). When any of them changes, calls ``build_report`` so the
served HTML always reflects current state. Polling avoids a watchdog dep and
works fine across NFS.

Run:
    python scripts/watch_and_rebuild.py --interval 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geodisaster.report import build_report   # noqa: E402
from geodisaster.blog import build_blog       # noqa: E402

WATCHED_GLOBS = (
    "outputs/few_shot_*/few_shot_results.csv",
    "outputs/sen1floods11_*comparison*.json",
    "outputs/sen1floods11_results_table.json",
    "outputs/four_way_results_table.json",
    "outputs/reproducibility.json",
    "outputs/figures/*.png",
    "outputs/active_adapt/*.json",
    "outputs/leave_one_region_out_multiseed/seed*/results.json",
    "outputs/zero_shot/*/flood_decision_summary.json",
    "data/catalog/*.yaml",
)


def _refresh_derived_figures():
    """Re-render derived figures (Fig 10 etc.) from their JSON sources
    before each blog rebuild, so live experiment outputs appear on the site."""
    import subprocess
    try:
        subprocess.run(
            ["python", "scripts/refresh_derived_figures.py"],
            capture_output=True, timeout=120,
        )
    except Exception:
        pass


def _snapshot(globs) -> dict[str, float]:
    snap: dict[str, float] = {}
    for g in globs:
        for p in Path(".").glob(g):
            try:
                snap[str(p)] = p.stat().st_mtime
            except OSError:
                pass
    return snap


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=5.0,
                   help="seconds between polls")
    p.add_argument("--out", default="outputs/site")
    args = p.parse_args()

    last = _snapshot(WATCHED_GLOBS)
    print(f"[watch] tracking {len(last)} files; rebuild every change "
          f"(poll every {args.interval}s). Ctrl-C to stop.")

    def _rebuild():
        _refresh_derived_figures()
        dash = build_report(out_dir=Path(args.out), filename="dashboard.html")
        blog = build_blog(out_path=Path(args.out) / "index.html")
        return dash, blog

    dash, blog = _rebuild()
    print(f"[watch] initial build: dashboard={dash} ({int(dash.stat().st_size/1024)} KB), "
          f"blog={blog} ({int(blog.stat().st_size/1024)} KB)")

    while True:
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("[watch] stopped.")
            return 0
        cur = _snapshot(WATCHED_GLOBS)
        added = set(cur) - set(last)
        removed = set(last) - set(cur)
        changed = {p for p in cur if p in last and cur[p] != last[p]}
        if added or removed or changed:
            ts = time.strftime("%H:%M:%S")
            print(f"[watch {ts}] +{len(added)} -{len(removed)} ~{len(changed)} files; rebuilding...")
            dash, blog = _rebuild()
            print(f"[watch {ts}] rebuilt dashboard={int(dash.stat().st_size/1024)} KB "
                  f"blog={int(blog.stat().st_size/1024)} KB")
            last = cur


if __name__ == "__main__":
    sys.exit(main())
