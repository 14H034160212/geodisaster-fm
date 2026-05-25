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

WATCHED_GLOBS = (
    "outputs/few_shot_*/few_shot_results.csv",
    "outputs/sen1floods11_*comparison*.json",
    "outputs/sen1floods11_results_table.json",
    "outputs/four_way_results_table.json",
    "outputs/reproducibility.json",
    "outputs/figures/*.png",
    "data/catalog/*.yaml",
)


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
    # Initial build so the page exists even if nothing has changed yet
    out = build_report(out_dir=args.out)
    print(f"[watch] initial build: {out}")

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
            out = build_report(out_dir=args.out)
            size_kb = int(Path(out).stat().st_size / 1024)
            print(f"[watch {ts}] rebuilt {out} ({size_kb} KB)")
            last = cur


if __name__ == "__main__":
    sys.exit(main())
