"""OSM road and building fetchers.

Uses `osmnx` against the Overpass API. Cached to ``data/external/osm/`` so we
don't re-hammer Overpass during dev. Outputs GeoPackage layers per event.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from .catalog import DisasterEvent
from ..utils.logging import get_logger

log = get_logger("osm")

DEFAULT_HIGHWAY_TYPES = (
    "motorway", "trunk", "primary", "secondary", "tertiary", "residential",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
)


def _bbox_tuple(event: DisasterEvent) -> tuple[float, float, float, float]:
    if event.bbox is None:
        raise ValueError(f"event {event.event_id} has no bbox")
    return event.bbox


def fetch_roads(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> Path:
    import osmnx as ox
    import geopandas as gpd

    minx, miny, maxx, maxy = _bbox_tuple(event)
    types = list(cfg.get("highway_types", list(DEFAULT_HIGHWAY_TYPES)))
    cf = '["highway"~"' + "|".join(types) + '"]'
    log.info("osm_fetch_roads", event=event.event_id, bbox=(minx, miny, maxx, maxy))

    g = ox.graph_from_bbox(maxy, miny, maxx, minx, custom_filter=cf, simplify=True)
    edges = ox.graph_to_gdfs(g, nodes=False)
    edges = edges.reset_index()[["u", "v", "key", "highway", "name", "length", "geometry"]]
    edges["highway"] = edges["highway"].astype(str)

    out_path = Path(out_dir) / f"{event.event_id}_roads.gpkg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_file(out_path, layer="roads", driver="GPKG")
    return out_path


def fetch_buildings(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> Path:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = _bbox_tuple(event)
    log.info("osm_fetch_buildings", event=event.event_id, bbox=(minx, miny, maxx, maxy))
    tags = {"building": True}
    poly = box(minx, miny, maxx, maxy)
    gdf = ox.features_from_polygon(poly, tags=tags)
    # Keep polygons only
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    keep_cols = [c for c in ("building", "name", "amenity", "height", "levels", "geometry") if c in gdf.columns]
    gdf = gdf[keep_cols]

    out_path = Path(out_dir) / f"{event.event_id}_buildings.gpkg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, layer="buildings", driver="GPKG")
    return out_path


def fetch_facilities(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> Path:
    """Critical facilities: hospitals, schools, evacuation shelters."""
    import osmnx as ox
    from shapely.geometry import box

    minx, miny, maxx, maxy = _bbox_tuple(event)
    tags = {
        "amenity": ["hospital", "clinic", "school", "kindergarten", "shelter"],
        "emergency": ["assembly_point"],
    }
    poly = box(minx, miny, maxx, maxy)
    gdf = ox.features_from_polygon(poly, tags=tags)
    gdf = gdf.reset_index(drop=True)

    out_path = Path(out_dir) / f"{event.event_id}_facilities.gpkg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, layer="facilities", driver="GPKG")
    return out_path
