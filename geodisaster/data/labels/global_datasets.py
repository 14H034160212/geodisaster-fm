"""Global public disaster benchmark ingestion.

Covers the three datasets called out in the proposal §7 / data-requirements §1
that we need for **global-to-Japan transfer** experiments:

- **xBD** (xView2): building damage classification from Maxar pre/post pairs.
  Registration required at https://xview2.org/.
- **OpenEarthMap**: 8-class land-cover with disaster-relevant classes.
  https://open-earth-map.org/  (download via HuggingFace Hub).
- **Sen1Floods11**: Sentinel-1 hand-labeled flood masks across 11 events.
  https://github.com/cloudtostreet/Sen1Floods11

These adapters do *not* download anything by themselves — every dataset has
licensing or registration in the way. They produce a normalized
``DatasetManifest`` (file paths, splits, classes) consumed by the dataset/
training code in P2/P3.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ...utils.logging import get_logger

log = get_logger("labels.global")


@dataclass
class DatasetManifest:
    name: str
    root: str  # absolute path on disk
    split: str  # train / val / test
    image_paths: list[str] = field(default_factory=list)
    label_paths: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "image_path": self.image_paths,
            "label_path": self.label_paths,
            "split": self.split,
        })


# ---------------------------------------------------------------------------
# xBD
# ---------------------------------------------------------------------------
def ingest_xbd(root: str | Path, split: str = "train") -> DatasetManifest:
    """xBD directory layout:
        root/
            train/images/   <event>_<id>_pre_disaster.png
            train/labels/   <event>_<id>_pre_disaster.json   (polygons + damage class)
            test/...
            hold/...
    """
    root = Path(root)
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"xBD images dir missing: {images_dir}. Register at xview2.org.")
    pre_imgs = sorted(images_dir.glob("*_pre_disaster.png"))
    pairs = []
    for pre in pre_imgs:
        stem = pre.name.replace("_pre_disaster.png", "")
        post = images_dir / f"{stem}_post_disaster.png"
        post_label = labels_dir / f"{stem}_post_disaster.json"
        if post.exists() and post_label.exists():
            pairs.append((str(pre), str(post), str(post_label)))
    log.info("xbd_ingested", split=split, pairs=len(pairs))
    return DatasetManifest(
        name="xbd",
        root=str(root),
        split=split,
        image_paths=[p[1] for p in pairs],  # post-disaster image as primary
        label_paths=[p[2] for p in pairs],
        classes=["no-damage", "minor-damage", "major-damage", "destroyed"],
        meta={"pre_images": [p[0] for p in pairs]},
    )


# ---------------------------------------------------------------------------
# OpenEarthMap
# ---------------------------------------------------------------------------
OEM_CLASSES = [
    "bareland", "rangeland", "developed_space", "road",
    "tree", "water", "agriculture_land", "building",
]


def ingest_openearthmap(root: str | Path, split: str = "train") -> DatasetManifest:
    """OpenEarthMap layout: ``root/<region>/images/*.tif`` and ``.../labels/*.tif``
    with split lists under ``root/list/`` (train.txt / val.txt).
    """
    root = Path(root)
    split_file = root / "list" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"OpenEarthMap split file missing: {split_file}")
    pairs = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # OEM convention: <region>/<filename>.tif lives in images/ and labels/
        rel = Path(line.strip())
        img = root / rel.parent / "images" / rel.name
        lbl = root / rel.parent / "labels" / rel.name
        if img.exists() and lbl.exists():
            pairs.append((str(img), str(lbl)))
    log.info("oem_ingested", split=split, pairs=len(pairs))
    return DatasetManifest(
        name="openearthmap",
        root=str(root),
        split=split,
        image_paths=[p[0] for p in pairs],
        label_paths=[p[1] for p in pairs],
        classes=OEM_CLASSES,
    )


# ---------------------------------------------------------------------------
# Sen1Floods11
# ---------------------------------------------------------------------------
def ingest_sen1floods11(root: str | Path, split: str = "train") -> DatasetManifest:
    """Sen1Floods11 layout: ``v1.1/data/flood_events/HandLabeled/`` with
    S1Hand (input) and LabelHand (label) subdirs; CSVs in ``splits/``.
    """
    root = Path(root)
    csv_dir = root / "v1.1" / "splits"
    sx = "flood_train_data.csv" if split == "train" else (
        "flood_valid_data.csv" if split == "val" else "flood_test_data.csv"
    )
    csv = csv_dir / sx
    if not csv.exists():
        raise FileNotFoundError(f"Sen1Floods11 split CSV missing: {csv}")
    df = pd.read_csv(csv, header=None, names=["image", "label"])
    base = root / "v1.1" / "data" / "flood_events" / "HandLabeled"
    img_paths = [str(base / "S1Hand" / r.image) for _, r in df.iterrows()]
    lbl_paths = [str(base / "LabelHand" / r.label) for _, r in df.iterrows()]
    log.info("sen1floods11_ingested", split=split, pairs=len(df))
    return DatasetManifest(
        name="sen1floods11",
        root=str(root),
        split=split,
        image_paths=img_paths,
        label_paths=lbl_paths,
        classes=["non_water", "water"],
        meta={"sensors": ["S1"], "polarizations": ["VV", "VH"]},
    )
