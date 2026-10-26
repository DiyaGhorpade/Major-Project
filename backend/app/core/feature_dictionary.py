"""Display names, units and plain-language descriptions for the ML classifier's feature names.

The ML service reports raw feature names such as `soil_roll6_mean`. They follow a small set of patterns
(`feature_engineering.py` in the ML package: sensor x {lag1, diff1, trend3, rollN_mean/std, base_dev, longN_mean,
longN_dev, ref_dev, ref_dev_rollN}, derived and time features), so one pattern table covers every name listed in
`classifier.meta.json`. Text only
describes how a feature is computed; it never claims a biological cause. `resolve` reports whether a name
matched a known pattern (`known=False` means a generic fallback was used).
"""

import re
from dataclasses import dataclass

from app.core.units import format_value


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    display_name: str
    unit: str  # "" when the feature has no natural unit
    description: str
    known: bool = True


# sensor key -> (label, unit)
_SENSORS: dict[str, tuple[str, str]] = {
    "soil": ("Soil moisture", "%"),
    "temperature": ("Temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "light": ("Light", "lux"),
    "vpd": ("Vapour pressure deficit", "kPa"),
    "log_light": ("Light (log scale)", "ln(lux)"),
}
_SENSOR_ALT = "|".join(sorted(_SENSORS, key=len, reverse=True))

_STATS = {"mean": "average", "std": "variability"}
_BASELINE_ROWS = 48  # ML config features.baseline_window

_TIME_FEATURES: dict[str, tuple[str, str, str]] = {
    "hour": ("Hour of day", "h", "Hour of the day (0-23, local time) of the reading."),
    "dayofweek": ("Day of week", "", "Day of the week of the reading (0 = Monday ... 6 = Sunday)."),
    "hour_sin": ("Time of day (sine)", "", "Sine of the hour of day; with the cosine it encodes the time of day on a 24-hour cycle."),
    "hour_cos": ("Time of day (cosine)", "", "Cosine of the hour of day; with the sine it encodes the time of day on a 24-hour cycle."),
}
_DERIVED: dict[str, tuple[str, str, str]] = {
    "light_soil_interaction": (
        "Light x soil moisture", "",
        "Product of the (scaled) log light level and the soil moisture, so the model can pick up combinations of the two.",
    ),
    "soil_rate_per_hr": (
        "Soil moisture change rate", "% per hour",
        "How fast soil moisture changed since this plant's previous reading, per hour.",
    ),
}

_ROLL = re.compile(rf"^({_SENSOR_ALT})_roll(\d+)_(mean|std)$")
_LAG = re.compile(rf"^({_SENSOR_ALT})_lag1$")
_DIFF = re.compile(rf"^({_SENSOR_ALT})_diff1$")
_TREND = re.compile(rf"^({_SENSOR_ALT})_trend3$")
_BASE = re.compile(rf"^({_SENSOR_ALT})_base_dev$")
_LONG_MEAN = re.compile(rf"^({_SENSOR_ALT})_long(\d+)_mean$")
_LONG_DEV = re.compile(rf"^({_SENSOR_ALT})_long(\d+)_dev$")
_REF = re.compile(rf"^({_SENSOR_ALT})_ref_dev$")
_REF_ROLL = re.compile(rf"^({_SENSOR_ALT})_ref_dev_roll(\d+)$")


def _info(name: str, display: str, unit: str, description: str) -> FeatureInfo:
    return FeatureInfo(name, display, unit, description)


def resolve(name: str) -> FeatureInfo:
    """Look a feature name up; unknown names get a generic (but honest) fallback with `known=False`."""
    if name in _SENSORS:
        label, unit = _SENSORS[name]
        return _info(name, label, unit, f"The {label.lower()} measured in this reading.")
    if name in _DERIVED:
        label, unit, text = _DERIVED[name]
        return _info(name, label, unit, text)
    if name in _TIME_FEATURES:
        label, unit, text = _TIME_FEATURES[name]
        return _info(name, label, unit, text)

    if m := _ROLL.match(name):
        sensor, n, stat = m.groups()
        label, unit = _SENSORS[sensor]
        if stat == "mean":
            return _info(name, f"{label}, {n}-reading average", unit,
                         f"Average {label.lower()} over this plant's last {n} readings.")  # fmt: skip
        return _info(name, f"{label}, {n}-reading variability", unit,
                     f"How much {label.lower()} varied (standard deviation) over this plant's last {n} readings.")  # fmt: skip
    if m := _LAG.match(name):
        label, unit = _SENSORS[m.group(1)]
        return _info(name, f"{label}, previous reading", unit, f"{label} at this plant's previous reading.")
    if m := _DIFF.match(name):
        label, unit = _SENSORS[m.group(1)]
        return _info(name, f"{label}, change since previous reading", unit,
                     f"Change in {label.lower()} compared with this plant's previous reading.")  # fmt: skip
    if m := _TREND.match(name):
        label, unit = _SENSORS[m.group(1)]
        return _info(name, f"{label}, 3-reading change", unit,
                     f"Change in {label.lower()} compared with 3 readings earlier.")  # fmt: skip
    if m := _BASE.match(name):
        label, unit = _SENSORS[m.group(1)]
        return _info(name, f"{label}, difference from usual", unit,
                     f"How far {label.lower()} is from this plant's own recent baseline "
                     f"(the average of its previous {_BASELINE_ROWS} readings).")  # fmt: skip

    if m := _LONG_MEAN.match(name):
        sensor, n = m.groups()
        label, unit = _SENSORS[sensor]
        return _info(name, f"{label}, {n}-reading average", unit,
                     f"Average {label.lower()} over this plant's last {n} readings.")  # fmt: skip
    if m := _LONG_DEV.match(name):
        sensor, n = m.groups()
        label, unit = _SENSORS[sensor]
        return _info(name, f"{label}, difference from {n}-reading usual", unit,
                     f"How far {label.lower()} is from the average of this plant's previous {n} readings.")  # fmt: skip
    if m := _REF.match(name):
        label, unit = _SENSORS[m.group(1)]
        return _info(name, f"{label} vs. control plants", unit,
                     f"How far this plant's {label.lower()} is from the median of the healthy-control plants "
                     "measured at the same time (the shared room conditions cancel out).")  # fmt: skip
    if m := _REF_ROLL.match(name):
        sensor, n = m.groups()
        label, unit = _SENSORS[sensor]
        return _info(name, f"{label} vs. control plants, {n}-reading average", unit,
                     f"Average, over this plant's last {n} readings, of how far its {label.lower()} was from the "
                     "healthy-control plants measured at the same time.")  # fmt: skip

    pretty = name.replace("_", " ").strip().capitalize() or "Unknown feature"
    return FeatureInfo(name, pretty, "", "A model input derived from the sensor readings.", known=False)


def local_explanation(info: FeatureInfo, direction: str, target_class: str, value: float | None) -> str:
    """One plain sentence about a single local contribution, using only the stored numbers."""
    seen = f"was {format_value(value, info.unit)}" if value is not None else "had no measured value"
    if direction == "increases_prediction":
        effect = f"pushed the model towards '{target_class}'"
    elif direction == "decreases_prediction":
        effect = f"pushed the model away from '{target_class}'"
    else:
        effect = f"did not move the model's score for '{target_class}'"
    return f"{info.display_name} {seen} and {effect}."


def global_interpretation(info: FeatureInfo, mean_abs: float, rank: int, sample_count: int) -> str:
    return (
        f"Rank {rank}: on average it moved the model's score by about {mean_abs:.2f} (log-odds) per explained "
        f"row, across {sample_count} test rows and all four classes. This describes the model, not a cause in the plant."
    )


def global_tip(info: FeatureInfo) -> str:
    return f"{info.description} Larger bars mean the model relies on this input more."
