"""Static Tulsi plant catalog for plants 1-16.

The dataset carries no plant names, species, descriptions, locations or optimal ranges, so
they live here (user decision: keep the Tulsi branding). The copy is neutral and condition-independent: it
describes the variety and the pot's role in the experiment, never how the plant is doing. Odd plants are Rama Tulsi (green), even plants
Krishna Tulsi (purple). The experimental group (Cluster 1-4) comes from `mappings`; Cluster 4
is a temperature/humidity trial, not a nutrient trial.
"""

from dataclasses import dataclass
from typing import Any

from app.core.mappings import GroupInfo, group_for_plant

RAMA = ("Rama Tulsi (Green)", "Ocimum sanctum")
KRISHNA = ("Krishna Tulsi (Purple)", "Ocimum tenuiflorum")

LOCATIONS = {
    "Cluster 1": "Bay 1 — Control Bench",
    "Cluster 2": "Bay 2 — Watering Trial Row",
    "Cluster 3": "Bay 3 — Light Trial Box",
    "Cluster 4": "Bay 4 — Climate Trial Row",
}

# Name suffix per group: the experimental role of the pot, not a claim about its condition.
_NAME_SUFFIX = {
    "Cluster 1": "Control Group",
    "Cluster 2": "Watering Trial",
    "Cluster 3": "Light Trial",
    "Cluster 4": "Climate Trial",
}

# Neutral, condition-independent copy. Nothing here says how a plant is doing: the dataset has no such
# information and health statements must come only from model output (hard rule 4).
_VARIETY_COPY = {
    "Rama": "Rama Tulsi (Ocimum sanctum) is the green-leaved variety.",
    "Krishna": "Krishna Tulsi (Ocimum tenuiflorum) is the purple-leaved variety.",
}
_GROUP_COPY = {  # (what the group is in the experiment, generic care guidance)
    "Cluster 1": (
        "Part of the control group, kept under the standard conditions the other groups are compared with.",
        "Water when the top of the soil feels dry, avoid waterlogging, and give bright light.",
    ),
    "Cluster 2": (
        "Part of the watering trial, where the watering routine is varied on purpose.",
        "Check the soil by hand before watering and follow the trial's watering plan; avoid waterlogging.",
    ),
    "Cluster 3": (
        "Part of the light trial, where the amount of light is varied on purpose.",
        "Follow the trial's lighting plan and keep the sensor and plant positions unchanged.",
    ),
    "Cluster 4": (
        "Part of the climate trial, where temperature and humidity are varied on purpose.",
        "Follow the trial's climate plan and keep the pot away from direct heat sources and draughts.",
    ),
}


def _range(lo: float, hi: float, unit: str, ideal: str, low: str, high: str) -> dict[str, Any]:
    sep = "" if unit in ("%", "°C") else " "
    u = f"{sep}{unit}"
    return {
        "min": lo,
        "max": hi,
        "unit": unit,
        "lowLabel": f"{low} (<{lo:,.0f}{u})",
        "idealLabel": f"{ideal} ({lo:,.0f}–{hi:,.0f}{u})",
        "highLabel": f"{high} (>{hi:,.0f}{u})",
    }


# Lower bounds mirror the ML service's recommendation thresholds (config.yaml recommendations:
# soil_low_pct 25, light_low_lux 300, light_high_lux 20000, temp_low_c 18, temp_high_c 32,
# humidity_low_pct 40). The ML config has no upper bound for soil or humidity, so the upper
# bounds below are plain horticultural choices for Tulsi: waterlogging risk above ~75 % soil
# moisture, and 90 % relative humidity as the point where air becomes persistently damp (the
# healthy control plants average ~78 %). Night-time light near 0 lux is normal, not a fault.
OPTIMAL_RANGES: dict[str, dict[str, Any]] = {
    "soil": _range(25, 75, "%", "Ideal", "Too Dry", "Too Wet"),
    "temperature": _range(18, 32, "°C", "Comfortable", "Cold", "Hot"),
    "humidity": _range(40, 90, "%", "Comfortable", "Dry Air", "Very Humid"),
    "light": _range(300, 20000, "lux", "Bright", "Dim", "Intense"),
}


METRICS = (  # (reading attribute = OPTIMAL_RANGES key, display label)
    ("soil", "Soil moisture"),
    ("temperature", "Temperature"),
    ("humidity", "Humidity"),
    ("light", "Light"),
)


@dataclass(frozen=True)
class PlantCatalogEntry:
    id: int
    name: str
    species: str
    variety: str
    group: GroupInfo
    description: str
    simple_care_tip: str
    location: str
    optimal_ranges: dict[str, Any]


def catalog_entry(plant_id: int) -> PlantCatalogEntry:
    group = group_for_plant(plant_id)
    variety, species = RAMA if plant_id % 2 == 1 else KRISHNA
    short = "Rama" if plant_id % 2 == 1 else "Krishna"
    suffix = _NAME_SUFFIX[group.cluster]
    role, tip = _GROUP_COPY[group.cluster]
    description = f"{_VARIETY_COPY[short]} {role}"
    return PlantCatalogEntry(
        id=plant_id,
        name=f"Pot {plant_id}: {short} Tulsi ({suffix})",
        species=species,
        variety=variety,
        group=group,
        description=description,
        simple_care_tip=tip,
        location=LOCATIONS[group.cluster],
        optimal_ranges=OPTIMAL_RANGES,
    )


PLANT_IDS = tuple(range(1, 17))
