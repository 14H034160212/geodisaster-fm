"""Figure 2 — Japan multi-hazard event coverage map.

Plots event AOIs over a Japan basemap, colored by hazard type.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ..data.catalog import EventCatalog, HazardType


HAZARD_COLORS = {
    HazardType.FLOOD:      "#1f77b4",
    HazardType.LANDSLIDE:  "#d62728",
    HazardType.EARTHQUAKE: "#9467bd",
    HazardType.TSUNAMI:    "#17becf",
    HazardType.TYPHOON:    "#ff7f0e",
    HazardType.VOLCANIC:   "#8c564b",
    HazardType.COMPOUND:   "#2ca02c",
}


def render(catalog: EventCatalog, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(8, 9))
        ax = plt.axes(projection=proj)
        ax.set_extent([122, 153, 24, 46], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor="#f3f3f3")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.set_xlim(122, 153); ax.set_ylim(24, 46)
        ax.set_facecolor("#f3f3f3")

    for e in catalog:
        if e.bbox is None:
            continue
        minx, miny, maxx, maxy = e.bbox
        color = HAZARD_COLORS.get(e.hazard, "gray")
        ax.add_patch(mpatches.Rectangle(
            (minx, miny), maxx - minx, maxy - miny,
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.35,
        ))
        ax.text((minx + maxx) / 2, miny - 0.15, e.event_id.split("_")[-1],
                fontsize=6.5, ha="center", color=color)

    legend = [
        mpatches.Patch(color=c, label=h.value)
        for h, c in HAZARD_COLORS.items()
        if any(e.hazard == h for e in catalog)
    ]
    ax.legend(handles=legend, loc="lower right", title="hazard", fontsize=8)
    ax.set_title("GeoDisaster-FM event coverage (Japan)", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
