"""Standard emergency-response question set.

These are the ten questions a UN OCHA / IFRC / national civil-protection
desk asks within the first 24 hours of a flood event. Each query is
formulated against the same graph state: ``G`` = passable-road graph
(flooded edges removed), ``F`` = facility nodes (hospitals, schools,
shelters), ``B`` = building polygons, ``P`` = population raster.

The Layer 3 RL policy will eventually decide *which* of these to answer
first under time pressure; Layer 2 just answers them all in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EmergencyQuery:
    """One of the ten standard emergency questions."""

    qid: str
    text: str
    category: str            # "exposure" / "access" / "priority"
    answer_kind: str         # "count" / "list" / "ranked_list" / "km"
    description: str         # how the reasoner answers it


STANDARD_QUERIES: list[EmergencyQuery] = [
    EmergencyQuery(
        qid="Q1",
        text="Which hospitals are inside the flood footprint?",
        category="exposure",
        answer_kind="list",
        description="Intersect OSM amenity=hospital with the flood mask.",
    ),
    EmergencyQuery(
        qid="Q2",
        text="Which schools / shelters are inside the flood footprint?",
        category="exposure",
        answer_kind="list",
        description="Intersect OSM amenity=school/shelter with the flood mask.",
    ),
    EmergencyQuery(
        qid="Q3",
        text="How many buildings are predicted affected?",
        category="exposure",
        answer_kind="count",
        description="OSM building polygons with >=20% mask intersection.",
    ),
    EmergencyQuery(
        qid="Q4",
        text="How many km of major roads are blocked?",
        category="exposure",
        answer_kind="km",
        description="OSM major road segments with >=15% mask intersection.",
    ),
    EmergencyQuery(
        qid="Q5",
        text="What populated areas are now disconnected from any hospital?",
        category="access",
        answer_kind="ranked_list",
        description=(
            "After removing flooded road segments, find connected components "
            "of the road graph containing populated nodes but no hospital, "
            "ranked by population."
        ),
    ),
    EmergencyQuery(
        qid="Q6",
        text="Which hospitals lose road access from their service area?",
        category="access",
        answer_kind="list",
        description=(
            "For each hospital h, check if its 10-km original service area "
            "still has connectivity to h after flooded edges are removed."
        ),
    ),
    EmergencyQuery(
        qid="Q7",
        text="Which top-5 roads, if cleared, restore the most access?",
        category="priority",
        answer_kind="ranked_list",
        description=(
            "Greedy marginal-utility ranking over flooded edges by "
            "population reconnected when each edge is restored."
        ),
    ),
    EmergencyQuery(
        qid="Q8",
        text="What is the total population in the flood footprint?",
        category="exposure",
        answer_kind="count",
        description=(
            "WorldPop raster summed over predicted-flooded pixels. (Requires "
            "a WorldPop GeoTIFF in the event AOI; if unavailable, returns "
            "None and falls back to OSM building counts.)"
        ),
    ),
    EmergencyQuery(
        qid="Q9",
        text="What is the population in disconnected communities?",
        category="priority",
        answer_kind="count",
        description="Sum of populations across components answered in Q5.",
    ),
    EmergencyQuery(
        qid="Q10",
        text="What is the time-to-evacuation estimate for affected populations?",
        category="priority",
        answer_kind="count",
        description=(
            "Heuristic: median shortest-path length (km) from population "
            "centroids in disconnected components to nearest still-reachable "
            "facility, ÷ 5 km/h walking speed. Reported in minutes."
        ),
    ),
]
