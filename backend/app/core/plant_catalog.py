"""Static Tulsi plant catalog for plants 1-16.

The dataset carries no plant names, species, descriptions, locations or optimal ranges, so
they live here (user decision: keep the Tulsi branding; copy adapted from the frontend mock
copy, minus its invented numeric readings). Odd plants are Rama Tulsi (green), even plants
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
    "Cluster 2": "Bay 2 — Drought Trial Row",
    "Cluster 3": "Bay 3 — Low-Light Shaded Box",
    "Cluster 4": "Bay 4 — Heat & Humidity Trial Row",
}

# Name suffix per group, as in the original frontend copy.
_NAME_SUFFIX = {
    "Cluster 1": {"Rama": "Green Leaf", "Krishna": "Purple Leaf"},
    "Cluster 2": {"Rama": "Thirsty / Dry", "Krishna": "Thirsty / Dry"},
    "Cluster 3": {"Rama": "Shaded / Dim", "Krishna": "Shaded / Dim"},
    "Cluster 4": {"Rama": "Heat / Humidity Trial", "Krishna": "Heat / Humidity Trial"},
}

# (description, simple care tip) per plant id.
_COPY: dict[int, tuple[str, str]] = {
    1: ("Control specimen grown under ideal room conditions. Vibrant green leaves with a rich clove-like aroma and active new shoot growth.",
        "Keep doing what you are doing. Steady moisture and plenty of light."),
    2: ("Anthocyanin-rich dark purple leaf variety. High eugenol essential oil concentration and a balanced vegetative canopy.",
        "Steady watering and bright indirect light keep the purple leaves vibrant."),
    3: ("Control group reference pot. Uniform stem node distance and strong natural resilience.",
        "Healthy and thriving. No action needed."),
    4: ("Control pot. Demonstrates steady daytime transpiration and responsive leaf stomata.",
        "Optimal condition. Keep the soil lightly moist."),
    5: ("Thirsty pot in the drought trial. Soil moisture is deliberately allowed to fall, so leaf stems may lose firmness (wilting).",
        "Watch the soil moisture. If it stays low, water until it just drains from the bottom."),
    6: ("Drought test pot. Lower leaves may soften and droop as moisture depletes.",
        "Watch the soil moisture. Water thoroughly if it keeps dropping."),
    7: ("Water stress research pot in the drought trial row.",
        "Check the soil. Restoring water protects the root hairs from drying out."),
    8: ("Drought test pot where curling leaf margins can appear as moisture is lost.",
        "Moderate watering usually perks the leaves back up within a few hours."),
    9: ("Kept in deep shade as part of the low-light trial. Stems may stretch tall and thin (etiolation) while seeking light.",
        "Needs more light. Move closer to a grow light or a bright window."),
    10: ("Light deficiency pot. New leaves may fade to dull green because purple anthocyanin pigment needs sunlight to develop.",
         "Needs more light. Several hours of bright light help dark purple leaves."),
    11: ("Light deprivation trial. Weak stem growth and a pale chlorophyll tone can result from lack of light.",
         "Needs more light. Switch on a full-spectrum LED overhead."),
    12: ("Light deficiency test pot. Water uptake can be slow because the leaves lack light to drive transpiration.",
         "Provide bright light for 10-12 hours daily."),
    13: ("Temperature and humidity trial pot. Conditions here are deliberately pushed away from the comfortable range.",
         "Check the climate in the bay. Keep the pot away from heat sources and avoid prolonged very dry or very damp air."),
    14: ("Heat and humidity trial pot. Leaf expansion can slow when air conditions swing widely.",
         "Check the climate in the bay and stabilise temperature and humidity."),
    15: ("Climate stress trial pot. Leaf edges may scorch under heat or sag in unusually damp air.",
         "Check temperature and humidity; ventilate if the air is stuffy or too dry."),
    16: ("Temperature and humidity trial pot. The plant is hydrated, but unstable air conditions can stress it.",
         "Check the climate in the bay and keep conditions steadier."),
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
    suffix = _NAME_SUFFIX[group.cluster][short]
    description, tip = _COPY[plant_id]
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
