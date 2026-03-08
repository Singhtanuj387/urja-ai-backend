from __future__ import annotations

from typing import Dict

# India's average grid emission factor (kg CO2 per kWh)
EMISSION_FACTOR_KG_PER_KWH: float = 0.82


def calculate_co2(kwh: float) -> float:
    """
    Calculate CO2 emissions in kilograms for a given energy consumption.
    """
    if kwh < 0:
        raise ValueError("kwh must be non-negative")
    return round(kwh * EMISSION_FACTOR_KG_PER_KWH, 3)


def calculate_co2_breakdown(kwh: float) -> Dict[str, float]:
    """
    Convenience helper that returns both energy and emissions.
    """
    co2_kg = calculate_co2(kwh)
    return {"energy_kwh": round(kwh, 3), "co2_kg": co2_kg}


__all__ = ["EMISSION_FACTOR_KG_PER_KWH", "calculate_co2", "calculate_co2_breakdown"]

