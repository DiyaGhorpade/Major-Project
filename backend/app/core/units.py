"""The one rounding rule for numbers shown in generated text (recommendations, SHAP sentences)."""


def format_value(value: float | None, unit: str = "") -> str:
    """`21.3 %`, `28.4 °C`, `1,234 lux`, `0.46 kPa`; `n/a` when missing.

    Decimals follow the unit: lux 0 (with thousands separators), % and °C 1, anything else 2.
    """
    if value is None:
        return "n/a"
    if unit == "lux":
        text = f"{value:,.0f}"
    elif unit in ("%", "°C"):
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return f"{text} {unit}".strip()
