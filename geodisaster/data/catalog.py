"""Disaster event catalog.

A DisasterEvent ties together an event name, hazard type, official source, time
window (pre/post), AOI bbox, and metadata required to drive every downstream
data pipeline (GEE downloads, label processing, dataset tiling, splits).

Catalogs are stored as YAML in ``data/catalog/`` so they are easy to inspect,
diff, and version. The schema deliberately mirrors the table in
``Nature_GeoFM_Disaster_Data_Requirements.docx`` §6.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import yaml


class HazardType(str, Enum):
    FLOOD = "flood"
    LANDSLIDE = "landslide"
    EARTHQUAKE = "earthquake"
    TSUNAMI = "tsunami"
    TYPHOON = "typhoon"
    VOLCANIC = "volcanic"
    COMPOUND = "compound"


@dataclass
class DisasterEvent:
    """One disaster event manifest entry. All fields match data-requirements §6."""

    event_id: str
    name: str
    hazard: HazardType
    country: str = "JP"
    region: str = ""  # 都道府県/市町村 or basin
    bbox: tuple[float, float, float, float] | None = None  # minx, miny, maxx, maxy (EPSG:4326)
    event_date: date | None = None
    pre_window: tuple[date, date] | None = None
    post_window: tuple[date, date] | None = None
    sources: list[str] = field(default_factory=list)  # GSI / JAXA / MLIT / paper / OSM ...
    label_paths: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hazard"] = self.hazard.value
        for k in ("event_date", "pre_window", "post_window"):
            v = getattr(self, k)
            if v is None:
                d[k] = None
            elif isinstance(v, date):
                d[k] = v.isoformat()
            else:
                d[k] = [v[0].isoformat(), v[1].isoformat()]
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> "DisasterEvent":
        def _parse_date(s):
            if s is None:
                return None
            return date.fromisoformat(s) if isinstance(s, str) else s

        def _parse_window(w):
            if w is None:
                return None
            return (_parse_date(w[0]), _parse_date(w[1]))

        return cls(
            event_id=raw["event_id"],
            name=raw["name"],
            hazard=HazardType(raw["hazard"]),
            country=raw.get("country", "JP"),
            region=raw.get("region", ""),
            bbox=tuple(raw["bbox"]) if raw.get("bbox") else None,
            event_date=_parse_date(raw.get("event_date")),
            pre_window=_parse_window(raw.get("pre_window")),
            post_window=_parse_window(raw.get("post_window")),
            sources=list(raw.get("sources", [])),
            label_paths=dict(raw.get("label_paths", {})),
            notes=raw.get("notes", ""),
        )


@dataclass
class EventCatalog:
    """A collection of DisasterEvent records with persistence + filtering."""

    events: list[DisasterEvent] = field(default_factory=list)

    def __iter__(self) -> Iterator[DisasterEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def add(self, event: DisasterEvent) -> None:
        if any(e.event_id == event.event_id for e in self.events):
            raise ValueError(f"duplicate event_id: {event.event_id}")
        self.events.append(event)

    def filter(
        self,
        hazard: HazardType | Iterable[HazardType] | None = None,
        region: str | None = None,
        country: str | None = None,
    ) -> "EventCatalog":
        hazards = None
        if hazard is not None:
            hazards = {hazard} if isinstance(hazard, HazardType) else set(hazard)

        def _ok(e: DisasterEvent) -> bool:
            if hazards is not None and e.hazard not in hazards:
                return False
            if region is not None and region.lower() not in e.region.lower():
                return False
            if country is not None and e.country != country:
                return False
            return True

        return EventCatalog([e for e in self.events if _ok(e)])

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.as_dict() for e in self.events])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"events": [e.as_dict() for e in self.events]}
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EventCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        events = [DisasterEvent.from_dict(e) for e in raw.get("events", [])]
        return cls(events=events)
