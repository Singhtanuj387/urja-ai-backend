from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import requests

_DEFAULT_GRID_API_URL = (
    "https://npp.gov.in/dashBoard/demandmet1chartdata?date=2026-03-03"
)


def _get_grid_api_url() -> str:
    """Read grid API URL from env, fallback to default."""
    return os.getenv("GRID_API_URL", _DEFAULT_GRID_API_URL)


def _fetch_grid_series() -> List[Dict[str, int]]:
    """
    Fetch raw grid demand data from the National Power Portal API.

    The API returns a list of objects with:
    - updated_on: epoch milliseconds
    - value_of_data: instantaneous demand (MW)
    """
    url = _get_grid_api_url()
    response = requests.get(url, timeout=8)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected grid API response format.")
    return data


def get_grid_load(hour: int) -> Dict[str, int]:
    """
    Derive a relative grid load percentage for a given hour of the day.

    Grid load is computed from historical demand data from the
    National Power Portal API by:
    - bucketing points by hour of day, averaging demand within the hour
    - scaling the hourly average against the daily maximum to get 0–100%
    """
    if not 0 <= hour <= 23:
        raise ValueError("hour must be in range [0, 23]")

    try:
        series = _fetch_grid_series()

        hourly_values: Dict[int, List[float]] = defaultdict(list)
        for entry in series:
            ts_ms = int(entry.get("updated_on", 0))
            demand = float(entry.get("value_of_data", 0))
            # Interpret timestamp in UTC; for a prototype this is sufficient
            dt = datetime.utcfromtimestamp(ts_ms / 1000.0)
            hourly_values[dt.hour].append(demand)

        if not hourly_values:
            raise ValueError("No hourly grid data available.")

        hourly_avg: Dict[int, float] = {
            h: (sum(vals) / len(vals)) for h, vals in hourly_values.items()
        }
        daily_peak = max(hourly_avg.values())
        if daily_peak <= 0:
            raise ValueError("Invalid peak demand from grid data.")

        if hour in hourly_avg:
            relative = hourly_avg[hour] / daily_peak
            grid_load = int(round(relative * 100))
        else:
            # If no data for this exact hour, fall back to a moderate load
            grid_load = 60

        grid_load = max(0, min(100, grid_load))
        return {"hour": hour, "grid_load": grid_load}
    except Exception:
        # Fallback: return a safe, moderate load if the external API is unavailable
        return {"hour": hour, "grid_load": 60}


__all__ = ["get_grid_load"]

