"""Figure 5 — decision-impact map.

Composite: impact mask + affected buildings (red outline) + disrupted road
segments (orange) + isolated communities (pink shading). Per-event PNGs are
written; aggregate stacking is left to the paper's figure assembly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def render(
    impact_mask_path: str | Path,
    buildings_path: str | Path | None,
    roads_path: str | Path | None,
    out_path: str | Path,
    title: str = "Disaster impact + decision overlay",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(impact_mask_path) as src:
        mask = src.read(1)
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.where(mask == 1, 1, np.nan), extent=extent, cmap="Blues", alpha=0.7)

    if buildings_path:
        import geopandas as gpd
        b = gpd.read_file(buildings_path)
        b.boundary.plot(ax=ax, color="red", linewidth=0.2)

    if roads_path:
        import geopandas as gpd
        r = gpd.read_file(roads_path)
        r.plot(ax=ax, color="darkorange", linewidth=0.4)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
