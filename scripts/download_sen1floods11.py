"""Download Sen1Floods11 HandLabeled set via anonymous gcsfs.

A no-gsutil-needed substitute. Pulls splits + S1Hand + S2Hand + LabelHand
into ``data/external/sen1floods11/v1.1/...`` in parallel.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import gcsfs


GCS_PREFIX = "sen1floods11/v1.1"
SUBDIRS = ("S1Hand", "S2Hand", "LabelHand")


def _download_one(fs, src: str, dst: Path) -> tuple[str, int]:
    if dst.exists() and dst.stat().st_size > 0:
        return ("skip", dst.stat().st_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fs.get_file(src, str(dst))
    return ("ok", dst.stat().st_size)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/external/sen1floods11")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--skip-s2", action="store_true",
                   help="skip Sentinel-2 chips (saves ~1 GB)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")

    # Splits
    splits_src = f"{GCS_PREFIX}/splits/flood_handlabeled"
    splits_dst = out / "v1.1" / "splits" / "flood_handlabeled"
    splits_dst.mkdir(parents=True, exist_ok=True)
    for entry in fs.ls(splits_src):
        name = entry.split("/")[-1]
        _download_one(fs, entry, splits_dst / name)
    print(f"  splits downloaded: {sorted(splits_dst.glob('*.csv'))}")

    subdirs = list(SUBDIRS)
    if args.skip_s2:
        subdirs.remove("S2Hand")

    jobs: list[tuple[str, Path]] = []
    for sub in subdirs:
        src_dir = f"{GCS_PREFIX}/data/flood_events/HandLabeled/{sub}"
        dst_dir = out / "v1.1" / "data" / "flood_events" / "HandLabeled" / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        entries = fs.ls(src_dir)
        for src in entries:
            name = src.split("/")[-1]
            jobs.append((src, dst_dir / name))
    print(f"  total chips queued: {len(jobs)} across {subdirs}")

    t0 = time.time()
    n_done = n_skip = 0
    total_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_download_one, fs, s, d) for s, d in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                kind, size = fut.result()
            except Exception as e:
                kind, size = ("err", 0)
                print(f"  err: {e}")
            if kind == "skip":
                n_skip += 1
            elif kind == "ok":
                n_done += 1
                total_bytes += size
            if i % 50 == 0 or i == len(futs):
                dt = time.time() - t0
                mb = total_bytes / 1e6
                rate = mb / dt if dt > 0 else 0
                print(f"  [{i:>4}/{len(futs)}] ok={n_done} skip={n_skip} "
                      f"{mb:.1f} MB in {dt:.1f}s ({rate:.1f} MB/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
