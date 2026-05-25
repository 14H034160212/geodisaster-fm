"""Sen1Floods11 -> GeoDisaster-FM patch format bridge.

Sen1Floods11 (Bonafilia et al. 2020) ships 446 hand-labeled 512x512 chips
across 11 flood events globally. The data is open and downloadable from
https://github.com/cloudtostreet/Sen1Floods11.

Layout we expect on disk (matches the v1.1 release):
    <root>/v1.1/
        data/flood_events/HandLabeled/
            S1Hand/<Region>_<id>_S1Hand.tif        # 2-band VV, VH (dB)
            LabelHand/<Region>_<id>_LabelHand.tif  # 0=non-water 1=water 255=no-data
        splits/
            flood_train_data.csv
            flood_valid_data.csv
            flood_test_data.csv

This converter does three things:
    1. Group chips by region (the filename prefix) — each region becomes a
       synthetic "event" so cross-event evaluation works.
    2. Save chip arrays as ``.npy`` with our standard ``__sentinel1.npy``,
       ``__label.npy`` naming so the existing DataModule + train CLI work.
    3. Emit ``data/catalog/sen1floods11_events.yaml`` so events show up in
       ``geodisaster list-events`` and can be used in train splits.

Optional: pass ``--with-alphaearth`` and we'll pull a 64-d AlphaEarth annual
embedding for each chip's bbox + the closest available year via GEE. This is
~hours for the full set (one GEE call per chip). Skip it for the first run.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..catalog import DisasterEvent, EventCatalog, HazardType
from ...utils.logging import get_logger

log = get_logger("sen1floods11")

REGION_PATTERN = re.compile(r"^([A-Za-z][A-Za-z\-]+)_\d+")

# Per-region approximate event dates from the Sen1Floods11 paper supplement.
# These power the AlphaEarth year selection when --with-alphaearth is on.
REGION_EVENT_YEAR: dict[str, int] = {
    "Bolivia":      2018,
    "Ghana":        2018,
    "India":        2016,
    "Mekong":       2018,
    "Nigeria":      2018,
    "Pakistan":     2017,
    "Paraguay":     2018,
    "Somalia":      2018,
    "Spain":        2019,
    "Sri-Lanka":    2017,
    "USA":          2016,
}


def _read_chip(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read a Sen1Floods11 chip, return (array, bbox_in_4326)."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        bounds = src.bounds
        crs = src.crs
        if crs is None or crs.to_string() == "EPSG:4326":
            bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        else:
            bbox = transform_bounds(crs, "EPSG:4326", *bounds)
    return arr, tuple(bbox)


def _region_of(filename: str) -> str:
    m = REGION_PATTERN.match(filename)
    return m.group(1) if m else "Unknown"


def _split_csv(root: Path, split: str) -> Path:
    fname = {
        "train":   "flood_train_data.csv",
        "val":     "flood_valid_data.csv",
        "test":    "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv",   # held-out Bolivia subset
    }[split]
    # Sen1Floods11 v1.1 stores splits under splits/flood_handlabeled/
    candidates = [
        root / "v1.1" / "splits" / "flood_handlabeled" / fname,
        root / "v1.1" / "splits" / fname,  # legacy layout
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def convert(
    sen1floods11_root: str | Path,
    out_patches: str | Path,
    out_catalog: str | Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    with_alphaearth: bool = False,
    gee_project: str | None = None,
) -> EventCatalog:
    sen1floods11_root = Path(sen1floods11_root)
    out_patches = Path(out_patches)
    out_patches.mkdir(parents=True, exist_ok=True)
    base = sen1floods11_root / "v1.1" / "data" / "flood_events" / "HandLabeled"

    if with_alphaearth:
        from ..gee import alphaearth as ae_mod, init_ee
        init_ee(project=gee_project)

    by_region: dict[str, list[dict]] = defaultdict(list)
    bbox_acc: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    split_assign: dict[str, str] = {}

    for split in splits:
        csv_path = _split_csv(sen1floods11_root, split)
        if not csv_path.exists():
            log.warning("missing_split", path=str(csv_path))
            continue
        df = pd.read_csv(csv_path, header=None, names=["image", "label"])
        log.info("ingesting_split", split=split, n=len(df))
        for _, row in df.iterrows():
            img_path = base / "S1Hand" / row.image
            lbl_path = base / "LabelHand" / row.label
            s2_path = base / "S2Hand" / row.image.replace("_S1Hand", "_S2Hand")
            if not (img_path.exists() and lbl_path.exists()):
                continue
            region = _region_of(img_path.name)
            try:
                s1, bbox = _read_chip(img_path)
                lbl, _ = _read_chip(lbl_path)
                lbl = lbl.astype(np.int16).squeeze()  # (H, W); -1=no-data in v1.1
                # Sen1Floods11 LabelHand uses -1 for no-data, 0=non-water, 1=water.
                # Re-encode to our convention (255 ignore index).
                lbl = np.where(lbl < 0, 255, lbl).astype(np.uint8)
            except Exception as e:
                log.warning("chip_read_failed", file=row.image, err=str(e))
                continue

            patch_id = f"{img_path.stem}"
            event_id = f"sen1floods11_{region}"
            event_dir = out_patches / event_id
            event_dir.mkdir(parents=True, exist_ok=True)

            s1_dst = event_dir / f"{patch_id}__sentinel1.npy"
            lbl_dst = event_dir / f"{patch_id}__label.npy"
            np.save(s1_dst, s1)
            np.save(lbl_dst, lbl)
            sources = {"sentinel1": str(s1_dst)}

            # Sen1Floods11 also ships Sentinel-2 chips; load if present.
            if s2_path.exists():
                try:
                    s2, _ = _read_chip(s2_path)
                    s2_dst = event_dir / f"{patch_id}__sentinel2.npy"
                    np.save(s2_dst, s2)
                    sources["sentinel2"] = str(s2_dst)
                except Exception as e:
                    log.warning("s2_read_failed", file=row.image, err=str(e))

            if with_alphaearth:
                try:
                    year = REGION_EVENT_YEAR.get(region, 2019) - 1  # truly pre-event
                    ae_arr = _fetch_alphaearth_chip(bbox, year, ae_mod, gee_project)
                    ae_dst = event_dir / f"{patch_id}__alphaearth.npy"
                    np.save(ae_dst, ae_arr)
                    sources["alphaearth"] = str(ae_dst)
                except Exception as e:
                    log.warning("ae_fetch_failed", patch=patch_id, err=str(e))

            valid = lbl != 255
            pos_frac = float((lbl == 1).sum() / max(valid.sum(), 1))
            by_region[region].append({
                "patch_id": patch_id,
                "row": 0, "col": 0, "size": int(lbl.shape[-1]),
                "sources": sources, "label_path": str(lbl_dst),
                "pos_fraction": pos_frac,
            })
            bbox_acc[region].append(bbox)
            split_assign[patch_id] = split

    # Emit per-event manifest.json
    for region, patches in by_region.items():
        event_id = f"sen1floods11_{region}"
        event_dir = out_patches / event_id
        manifest = {
            "event_id": event_id,
            "patch_size": patches[0]["size"] if patches else 512,
            "stride": patches[0]["size"] if patches else 512,
            "n_patches": len(patches),
            "ref_source": "sentinel1",
            "patches": patches,
            "split_assignment": {p["patch_id"]: split_assign.get(p["patch_id"], "train")
                                 for p in patches},
        }
        (event_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("region_written", region=region, n=len(patches))

    # Build a catalog covering the regions we materialized.
    events: list[DisasterEvent] = []
    for region, bboxes in bbox_acc.items():
        minx = min(b[0] for b in bboxes); miny = min(b[1] for b in bboxes)
        maxx = max(b[2] for b in bboxes); maxy = max(b[3] for b in bboxes)
        events.append(DisasterEvent(
            event_id=f"sen1floods11_{region}",
            name=f"Sen1Floods11 {region}",
            hazard=HazardType.FLOOD,
            country="XX",  # multi-country
            region=region,
            bbox=(minx, miny, maxx, maxy),
            sources=["Sen1Floods11 v1.1"],
            notes=f"Hand-labeled flood mask from Bonafilia et al. 2020. "
                  f"{len(by_region[region])} chips.",
        ))
    catalog = EventCatalog(events=events)
    catalog.save(out_catalog)
    log.info("catalog_written", path=str(out_catalog), n_events=len(events))
    return catalog


def _fetch_alphaearth_chip(bbox, year, ae_mod, gee_project):
    """Pull a 64-d AlphaEarth chip into a numpy array via GEE.

    Bypasses the file-based ExportSpec path and uses ``geemap.ee_to_numpy``
    directly so we don't litter disk with one-chip GeoTIFFs.
    """
    import ee
    import geemap
    coll = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    region = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    img = coll.filterDate(f"{year}-01-01", f"{year + 1}-01-01") \
              .filterBounds(region).mosaic().clip(region)
    arr = geemap.ee_to_numpy(img, region=region, scale=10)
    # geemap returns HWC; convert to CHW
    if arr is None:
        raise RuntimeError(f"AlphaEarth empty for bbox {bbox} year {year}")
    return np.transpose(arr.astype(np.float32), (2, 0, 1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("geodisaster ingest-sen1floods11")
    p.add_argument("--root", required=True, help="Sen1Floods11 v1.1 root dir")
    p.add_argument("--out-patches", default="data/processed/patches")
    p.add_argument("--out-catalog", default="data/catalog/sen1floods11_events.yaml")
    p.add_argument("--split", action="append", choices=["train", "val", "test"], default=None)
    p.add_argument("--with-alphaearth", action="store_true",
                   help="also pull a matching 64-d AlphaEarth chip per region (needs GEE auth)")
    p.add_argument("--gee-project", default=None)
    args = p.parse_args(argv)

    convert(
        sen1floods11_root=args.root,
        out_patches=args.out_patches,
        out_catalog=args.out_catalog,
        splits=tuple(args.split or ("train", "val", "test")),
        with_alphaearth=args.with_alphaearth,
        gee_project=args.gee_project,
    )
    return 0
