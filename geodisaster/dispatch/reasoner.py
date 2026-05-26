"""Layer 2 — Neuro-symbolic emergency reasoner.

Takes a flood-mask GeoTIFF + an AOI bbox and answers the ten standard
emergency questions defined in ``queries.py``. The graph reasoning is
pure symbolic (NetworkX + Shapely + Rasterio); the "neural" component is
the upstream pixel-level flood prediction produced by the perception
backbone (Layer 1, already trained).

Outputs a structured ``ReasonerReport`` (JSON-serialisable) and a
short natural-language briefing string.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask


def pd_concat(dfs):
    """Concat list of GeoDataFrames preserving CRS."""
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

from ..utils.logging import get_logger
from .queries import STANDARD_QUERIES, EmergencyQuery

log = get_logger("dispatch.reasoner")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class QueryAnswer:
    qid: str
    text: str
    answer: Any
    units: str = ""
    confidence: str = "high"   # high / medium / low
    note: str = ""


@dataclass
class ReasonerReport:
    event_id: str
    bbox: tuple[float, float, float, float]
    timestamp: str
    flood_mask_path: str
    n_buildings_in_aoi: int = 0
    n_facilities_in_aoi: int = 0
    road_km_in_aoi: float = 0.0
    answers: list[QueryAnswer] = field(default_factory=list)

    def briefing(self) -> str:
        """Render answers as a one-page text briefing — the responder's view."""
        lines = [
            f"=== Emergency briefing for {self.event_id} ===",
            f"  AOI bbox: {self.bbox}",
            f"  Building polygons in AOI:        {self.n_buildings_in_aoi:,}",
            f"  Critical facilities in AOI:      {self.n_facilities_in_aoi:,}",
            f"  Major road km in AOI:            {self.road_km_in_aoi:.1f}",
            "",
        ]
        for a in self.answers:
            ans = a.answer
            if isinstance(ans, list):
                if not ans:
                    val = "(none)"
                elif isinstance(ans[0], (str, int, float)):
                    val = ", ".join(str(x) for x in ans[:5]) + (
                        f"  (+{len(ans) - 5} more)" if len(ans) > 5 else "")
                else:
                    val = f"{len(ans)} items"
            elif isinstance(ans, dict):
                val = json.dumps(ans)
            else:
                val = str(ans)
            note = f"  [{a.note}]" if a.note else ""
            lines.append(f"  [{a.qid}] {a.text}")
            lines.append(f"        → {val} {a.units}{note}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------
class EmergencyReasoner:
    """Answer the ten standard emergency questions for one event."""

    def __init__(
        self,
        flood_mask_path: str | Path,
        bbox: tuple[float, float, float, float] | None = None,
        event_id: str = "unknown",
        worldpop_path: str | Path | None = None,
        permanent_water_path: str | Path | None = None,
    ):
        self.flood_mask_path = Path(flood_mask_path)
        self.event_id = event_id
        self.worldpop_path = Path(worldpop_path) if worldpop_path else None
        self.permanent_water_path = Path(permanent_water_path) if permanent_water_path else None

        with rasterio.open(self.flood_mask_path) as src:
            self._mask_bounds = src.bounds
            self._mask_crs = src.crs
        if bbox is None:
            bbox = (self._mask_bounds.left, self._mask_bounds.bottom,
                    self._mask_bounds.right, self._mask_bounds.top)
        self.bbox = bbox

    # ----- OSM ingestion -----
    def _fetch_osm(self, retries: int = 2, max_aoi_km: float = 15.0):
        """Fetch OSM data, automatically tiling AOIs larger than ``max_aoi_km``
        on a side. The default overpass-api.de mirror rate-limits research
        clusters quickly; we switch to kumi.systems which has higher quotas."""
        import math
        import time
        import geopandas as gpd
        import networkx as nx
        import osmnx as ox
        from shapely.geometry import box

        # Use kumi.systems Overpass mirror — main endpoint throttles quickly.
        try:
            ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        except Exception:
            pass

        minx, miny, maxx, maxy = self.bbox
        cy = 0.5 * (miny + maxy)
        deg_lat_per_km = 1.0 / 111.32
        deg_lon_per_km = 1.0 / (111.32 * max(math.cos(math.radians(cy)), 1e-6))
        # how many tiles on each axis so each tile is <= max_aoi_km
        nx_tiles = max(1, math.ceil((maxx - minx) / (max_aoi_km * deg_lon_per_km)))
        ny_tiles = max(1, math.ceil((maxy - miny) / (max_aoi_km * deg_lat_per_km)))
        tile_bboxes = []
        dx = (maxx - minx) / nx_tiles
        dy = (maxy - miny) / ny_tiles
        for ix in range(nx_tiles):
            for iy in range(ny_tiles):
                tx0 = minx + ix * dx
                ty0 = miny + iy * dy
                tile_bboxes.append((tx0, ty0, tx0 + dx, ty0 + dy))
        log.info("osm_tiling", n_tiles=len(tile_bboxes),
                 size_km=(max_aoi_km, max_aoi_km),
                 aoi_extent_deg=(round(maxx - minx, 3), round(maxy - miny, 3)))

        def _retry(fn, label, ti):
            last_err = None
            for attempt in range(retries):
                try:
                    return fn()
                except Exception as e:
                    last_err = e
                    time.sleep(1 + attempt)
            log.warning(f"osm_{label}_tile_failed", tile=ti,
                        err=str(last_err)[:100])
            return None

        b_dfs, f_dfs, sub_graphs, e_dfs = [], [], [], []
        facility_tags = {
            "amenity": ["hospital", "clinic", "school", "kindergarten", "shelter"],
            "emergency": ["assembly_point"],
        }
        cf = ('["highway"~"motorway|trunk|primary|secondary|tertiary|'
              'residential|unclassified"]')
        for ti, bbx in enumerate(tile_bboxes):
            tile_poly = box(*bbx)

            b = _retry(
                lambda p=tile_poly: ox.features_from_polygon(p, tags={"building": True}),
                "buildings", ti,
            )
            if b is not None and len(b):
                b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
                if len(b):
                    b_dfs.append(b)

            f = _retry(
                lambda p=tile_poly: ox.features_from_polygon(p, tags=facility_tags),
                "facilities", ti,
            )
            if f is not None and len(f):
                f_dfs.append(f.reset_index(drop=True))

            g_tile = _retry(
                lambda p=tile_poly: ox.graph_from_polygon(p, custom_filter=cf, simplify=True),
                "roads", ti,
            )
            if g_tile is not None and g_tile.number_of_edges() > 0:
                sub_graphs.append(g_tile)
                try:
                    e_tile = ox.graph_to_gdfs(g_tile, nodes=False).reset_index()
                    e_dfs.append(e_tile)
                except Exception:
                    pass

        buildings = (gpd.GeoDataFrame(pd_concat(b_dfs), crs="EPSG:4326")
                     if b_dfs else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        facilities = (gpd.GeoDataFrame(pd_concat(f_dfs), crs="EPSG:4326")
                      if f_dfs else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        edges = (gpd.GeoDataFrame(pd_concat(e_dfs), crs="EPSG:4326")
                 if e_dfs else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        # Merge per-tile graphs into one big graph
        if sub_graphs:
            g = nx.compose_all(sub_graphs)
        else:
            g = None
        log.info("osm_tile_aggregate",
                 n_tiles_used=sum(1 for x in sub_graphs),
                 buildings=int(len(buildings)),
                 facilities=int(len(facilities)),
                 edges=int(len(edges)))
        return buildings, facilities, g, edges

    # ----- raster helpers -----
    def _mask_zonal_fraction(self, geometries) -> np.ndarray:
        """For each geometry, return fraction of its mask=1 pixels."""
        out = np.zeros(len(geometries), dtype=np.float32)
        with rasterio.open(self.flood_mask_path) as src:
            for i, geom in enumerate(geometries):
                if geom is None or geom.is_empty:
                    continue
                try:
                    arr, _ = rio_mask(src, [geom.__geo_interface__], crop=True,
                                      all_touched=True, filled=False)
                    data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
                    if data.size:
                        out[i] = float((data == 1).mean())
                except Exception:
                    continue
        return out

    def _population_in_mask(self) -> float | None:
        """Sum WorldPop over flooded pixels. None if no pop raster."""
        if not (self.worldpop_path and self.worldpop_path.exists()):
            return None
        from rasterio.warp import reproject, Resampling
        with rasterio.open(self.flood_mask_path) as msk_src:
            mask = msk_src.read(1)
            mask_tf = msk_src.transform
            mask_crs = msk_src.crs
        with rasterio.open(self.worldpop_path) as pop_src:
            pop = np.zeros(mask.shape, dtype=np.float32)
            reproject(
                source=rasterio.band(pop_src, 1), destination=pop,
                src_transform=pop_src.transform, src_crs=pop_src.crs,
                dst_transform=mask_tf, dst_crs=mask_crs,
                resampling=Resampling.bilinear,
            )
        return float(pop[mask == 1].sum())

    # ----- main entry -----
    def run(self) -> ReasonerReport:
        import datetime as dt
        import networkx as nx

        log.info("reasoner_start", eid=self.event_id,
                 mask=str(self.flood_mask_path))
        buildings, facilities, g_full, edges = self._fetch_osm()
        log.info("osm_loaded", buildings=len(buildings),
                 facilities=len(facilities), edges=len(edges))

        # Project for length calculations
        edges_m = edges.to_crs(edges.estimate_utm_crs()) if len(edges) else edges
        total_road_km = float(edges_m.length.sum() / 1000.0) if len(edges) else 0.0

        report = ReasonerReport(
            event_id=self.event_id,
            bbox=self.bbox,
            timestamp=dt.datetime.utcnow().isoformat() + "Z",
            flood_mask_path=str(self.flood_mask_path),
            n_buildings_in_aoi=int(len(buildings)),
            n_facilities_in_aoi=int(len(facilities)),
            road_km_in_aoi=round(total_road_km, 1),
        )

        # Q1: hospitals in flood mask
        hosp = facilities[facilities.get("amenity").isin(["hospital", "clinic"])] \
                if "amenity" in facilities.columns else facilities.iloc[:0]
        affected_hosp: list[str] = []
        if len(hosp):
            hosp_buf = hosp.copy()
            hosp_buf["geometry"] = hosp_buf.geometry.buffer(0.0005)  # ~50m
            frac = self._mask_zonal_fraction(hosp_buf.geometry)
            for i, f in enumerate(frac):
                if f >= 0.2:
                    name = str(hosp.iloc[i].get("name", f"hospital_{i}"))
                    affected_hosp.append(name)
        report.answers.append(QueryAnswer(
            qid="Q1",
            text=STANDARD_QUERIES[0].text,
            answer=affected_hosp,
            units="hospitals",
        ))

        # Q2: schools/shelters in flood mask
        ss = facilities[facilities.get("amenity").isin(
            ["school", "kindergarten", "shelter"])] \
                if "amenity" in facilities.columns else facilities.iloc[:0]
        affected_ss: list[str] = []
        if len(ss):
            ss_buf = ss.copy()
            ss_buf["geometry"] = ss_buf.geometry.buffer(0.0005)
            frac = self._mask_zonal_fraction(ss_buf.geometry)
            for i, f in enumerate(frac):
                if f >= 0.2:
                    name = str(ss.iloc[i].get("name", f"school_{i}"))
                    affected_ss.append(name)
        report.answers.append(QueryAnswer(
            qid="Q2",
            text=STANDARD_QUERIES[1].text,
            answer=affected_ss,
            units="facilities",
        ))

        # Q3: affected building count
        affected_bld = 0
        if len(buildings):
            frac = self._mask_zonal_fraction(buildings.geometry)
            affected_bld = int((frac >= 0.2).sum())
        report.answers.append(QueryAnswer(
            qid="Q3",
            text=STANDARD_QUERIES[2].text,
            answer=affected_bld,
            units=f"of {len(buildings):,} buildings",
        ))

        # Q4: blocked road km
        blocked_km = 0.0
        edge_blocked: list[bool] = []
        if len(edges):
            frac = self._mask_zonal_fraction(edges.geometry)
            edge_blocked = [f >= 0.15 for f in frac]
            blocked_km = float(edges_m.length[edge_blocked].sum() / 1000.0)
        report.answers.append(QueryAnswer(
            qid="Q4",
            text=STANDARD_QUERIES[3].text,
            answer=round(blocked_km, 1),
            units=f"km of {total_road_km:.0f} km",
        ))

        # ----- Graph reasoning: Q5, Q6, Q7, Q9, Q10 -----
        # Build a passable graph by removing flooded edges
        components: list[dict] = []
        if g_full is not None and any(edge_blocked):
            # Map blocked status onto graph edges
            edge_index_map = {}
            for i, row in edges.iterrows():
                edge_index_map[(row["u"], row["v"], row["key"])] = i

            H = g_full.copy()
            to_remove = []
            for u, v, k in H.edges(keys=True):
                idx = edge_index_map.get((u, v, k)) or edge_index_map.get((v, u, k))
                if idx is not None and edge_blocked[idx]:
                    to_remove.append((u, v, k))
            H.remove_edges_from(to_remove)

            # Hospital nodes (snap to nearest road node)
            import osmnx as ox
            hosp_nodes: set = set()
            for _, h in hosp.iterrows():
                if h.geometry is None or h.geometry.is_empty:
                    continue
                pt = h.geometry.centroid
                try:
                    n = ox.distance.nearest_nodes(g_full, pt.x, pt.y)
                    hosp_nodes.add(n)
                except Exception:
                    continue

            # Connected components
            comps = list(nx.connected_components(H.to_undirected()))
            for cid, comp in enumerate(comps):
                has_hosp = bool(comp & hosp_nodes)
                # Population proxy: count building centroids in comp's nodes
                # convex hull (rough). Skip if too costly.
                comp_node_coords = [(g_full.nodes[n]["x"], g_full.nodes[n]["y"])
                                    for n in comp if "x" in g_full.nodes[n]]
                pop_proxy = len(comp_node_coords)   # crude
                components.append({
                    "component_id": cid,
                    "n_nodes": len(comp),
                    "has_hospital": has_hosp,
                    "node_proxy_population": pop_proxy,
                })

        # Q5: disconnected populated areas without a hospital
        disconnected = [c for c in components if not c["has_hospital"]]
        disconnected.sort(key=lambda c: -c["node_proxy_population"])
        top_disc = disconnected[:10]
        report.answers.append(QueryAnswer(
            qid="Q5",
            text=STANDARD_QUERIES[4].text,
            answer=top_disc,
            units="components",
            note=f"{len(disconnected)} components without lifeline",
            confidence="medium",
        ))

        # Q6: hospitals lose service-area road access (heuristic via component)
        hosp_isolated = []
        for cid, c in enumerate(components):
            if c["has_hospital"] and c["n_nodes"] < 20:
                hosp_isolated.append(f"hospital_in_component_{cid}")
        report.answers.append(QueryAnswer(
            qid="Q6",
            text=STANDARD_QUERIES[5].text,
            answer=hosp_isolated,
            units="hospitals with shrunk service area",
            confidence="medium",
        ))

        # Q7: top-5 roads to clear (placeholder ranked by blocked length)
        top_roads: list[dict] = []
        if len(edges) and any(edge_blocked):
            edges_with_len = edges_m.copy()
            edges_with_len["blocked"] = edge_blocked
            edges_with_len["len_m"] = edges_with_len.length
            top5 = (edges_with_len[edges_with_len["blocked"]]
                    .sort_values("len_m", ascending=False).head(5))
            for _, r in top5.iterrows():
                top_roads.append({
                    "highway": str(r.get("highway", "?")),
                    "name": str(r.get("name", "(unnamed)")),
                    "length_m": round(float(r["len_m"]), 1),
                })
        report.answers.append(QueryAnswer(
            qid="Q7",
            text=STANDARD_QUERIES[6].text,
            answer=top_roads,
            units="roads",
            note="ranked by length only; future RL refines",
            confidence="low",
        ))

        # Q8: total population in flood footprint
        pop_total = self._population_in_mask()
        report.answers.append(QueryAnswer(
            qid="Q8",
            text=STANDARD_QUERIES[7].text,
            answer=int(pop_total) if pop_total else None,
            units="people" if pop_total else "n/a (no WorldPop raster)",
            confidence="high" if pop_total else "low",
        ))

        # Q9: population in disconnected components (same proxy as Q5)
        pop_disc = sum(c["node_proxy_population"] for c in disconnected)
        report.answers.append(QueryAnswer(
            qid="Q9",
            text=STANDARD_QUERIES[8].text,
            answer=int(pop_disc),
            units="proxy (road-node count, not absolute pop)",
            confidence="low",
            note="needs WorldPop for absolute numbers",
        ))

        # Q10: time-to-evac heuristic
        # Median shortest-path from disconnected components to nearest
        # connected hospital node (5 km/h walking speed)
        eva_min = None
        if disconnected and hosp_nodes and g_full is not None:
            # Distances in graph metric (m), divide by walking speed
            sample = disconnected[:3]
            ds = []
            for c in sample:
                # rough sample size capped
                node_iter = list(comp_node_coords)[:5]
                ds.extend([1500.0] * len(node_iter))  # placeholder ~1.5km
            if ds:
                eva_min = round(float(np.median(ds)) / 1000.0 / 5.0 * 60.0, 0)
        report.answers.append(QueryAnswer(
            qid="Q10",
            text=STANDARD_QUERIES[9].text,
            answer=eva_min,
            units="minutes" if eva_min else "n/a",
            confidence="low",
            note="placeholder heuristic; needs Dijkstra over walking graph",
        ))

        log.info("reasoner_done", eid=self.event_id,
                 n_questions=len(report.answers))
        return report


def save_report(report: ReasonerReport, json_path: str | Path,
                briefing_path: str | Path | None = None) -> tuple[Path, Path | None]:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_json(), indent=2, default=str))
    if briefing_path:
        briefing_path = Path(briefing_path)
        briefing_path.write_text(report.briefing(), encoding="utf-8")
        return json_path, briefing_path
    return json_path, None
