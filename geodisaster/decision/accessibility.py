"""Post-disaster road accessibility and isolated-community analysis.

Pipeline (proposal §3 Task 3 / direction F):
    1. Build a road graph from OSM edges.
    2. Mark edges crossing the impact mask (fraction >= threshold) as DISRUPTED.
    3. Find connected components in the remaining graph.
    4. Identify settlements/populated places stuck in a component that no
       longer reaches a "lifeline" set (hospital, evacuation point, prefectural
       road core).
    5. Rank components by population to produce rescue priorities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IsolatedComponent:
    component_id: int
    n_nodes: int
    population: float
    contains_facility: bool


def _intersect_edges_with_mask(roads_gpkg: str | Path, mask_path: str | Path, threshold: float):
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask

    gdf = gpd.read_file(roads_gpkg)
    disrupted = np.zeros(len(gdf), dtype=bool)
    with rasterio.open(mask_path) as src:
        for i, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            buf = geom.buffer(0.00005)
            try:
                arr, _ = rio_mask(src, [buf.__geo_interface__], crop=True, all_touched=True, filled=False)
            except Exception:
                continue
            data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
            if data.size == 0:
                continue
            if (data == 1).mean() >= threshold:
                disrupted[i] = True
    return gdf, disrupted


def road_disruption_graph(roads_gpkg: str | Path, mask_path: str | Path, threshold: float = 0.2):
    """Build a NetworkX graph; mark disrupted edges as removed."""
    import networkx as nx

    gdf, disrupted = _intersect_edges_with_mask(roads_gpkg, mask_path, threshold)
    G = nx.MultiGraph()
    for i, row in gdf.iterrows():
        u = (round(row.geometry.coords[0][0], 6), round(row.geometry.coords[0][1], 6))
        v = (round(row.geometry.coords[-1][0], 6), round(row.geometry.coords[-1][1], 6))
        G.add_edge(u, v, key=i, length=row.geometry.length, disrupted=bool(disrupted[i]),
                   highway=str(row.get("highway", "unknown")))
    # Trim disrupted edges
    H = G.copy()
    H.remove_edges_from([(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d["disrupted"]])
    return G, H, gdf, disrupted


def isolated_communities(
    H,
    population_path: str | Path,
    facilities_gpkg: str | Path | None = None,
) -> list[IsolatedComponent]:
    """For each connected component in H, sum population around its nodes and
    flag the ones missing any lifeline facility.
    """
    import networkx as nx
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    import geopandas as gpd
    from shapely.geometry import Point

    facility_pts = []
    if facilities_gpkg:
        gdf = gpd.read_file(facilities_gpkg)
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Point":
                facility_pts.append((geom.x, geom.y))
            else:
                c = geom.centroid
                facility_pts.append((c.x, c.y))

    components = list(nx.connected_components(H))
    out: list[IsolatedComponent] = []
    with rasterio.open(population_path) as src:
        pop = src.read(1)
        tf = src.transform
        for cid, comp in enumerate(components):
            pop_total = 0.0
            has_facility = False
            for (x, y) in comp:
                r, c = rowcol(tf, x, y)
                if 0 <= r < pop.shape[0] and 0 <= c < pop.shape[1]:
                    pop_total += float(pop[r, c])
            if facility_pts:
                comp_set = comp
                for (fx, fy) in facility_pts:
                    nearest_node = min(comp_set, key=lambda n: (n[0] - fx) ** 2 + (n[1] - fy) ** 2)
                    dist2 = (nearest_node[0] - fx) ** 2 + (nearest_node[1] - fy) ** 2
                    if dist2 < (0.005) ** 2:  # ~500 m
                        has_facility = True
                        break
            out.append(IsolatedComponent(
                component_id=cid, n_nodes=len(comp), population=pop_total,
                contains_facility=has_facility,
            ))
    return out


def rescue_priority(components: list[IsolatedComponent]) -> list[IsolatedComponent]:
    """Sort: largest isolated population without a lifeline facility first."""
    return sorted(
        components,
        key=lambda c: (c.contains_facility, -c.population, -c.n_nodes),
    )
