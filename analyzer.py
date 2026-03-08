from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class DeviceUsageSession:
    """Represents a single ON–OFF usage window for a device."""

    device: str
    usage_start: datetime
    usage_end: datetime
    duration_minutes: int
    energy_kwh: float

    @property
    def usage_time_str(self) -> str:
        return f"{self.usage_start.strftime('%H:%M')}-{self.usage_end.strftime('%H:%M')}"


@dataclass
class DailyUsageSummary:
    """Aggregated usage statistics for a single day."""

    date: str
    total_energy_kwh: float
    peak_hour: int
    most_common_usage_hour: int
    hourly_energy_kwh: Dict[int, float]


def load_energy_data(path: str) -> pd.DataFrame:
    """
    Load time-series energy data from CSV or JSON.

    Expected columns: timestamp, device, power (W).
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Energy data file not found: {path}")

    if p.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Energy data file is empty.")

    if not {"timestamp", "device", "power"}.issubset(df.columns):
        raise ValueError("Data must contain 'timestamp', 'device', and 'power' columns.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _group_device_sessions(
    df: pd.DataFrame, power_threshold: float = 1.0
) -> List[DeviceUsageSession]:
    """
    Detect ON/OFF sessions for each device based on power readings.

    A simple rule-based detector:
    - A device is considered ON when power >= power_threshold.
    - Sessions are contiguous runs of ON readings.
    - Energy (kWh) = sum(P(W) * Δt(h)) over the session.
    """
    sessions: List[DeviceUsageSession] = []

    # Ensure we have per-minute or regular-ish sampling; compute delta between rows.
    df = df.copy()
    df["timestamp_shifted"] = df.groupby("device")["timestamp"].shift(1)
    df["delta_minutes"] = (
        (df["timestamp"] - df["timestamp_shifted"]).dt.total_seconds().fillna(0) / 60.0
    )

    for device, group in df.groupby("device"):
        group = group.sort_values("timestamp")

        current_on: bool = False
        start_idx: Optional[int] = None
        energy_wh: float = 0.0

        for idx, row in group.iterrows():
            power = float(row["power"])
            is_on = power >= power_threshold

            if is_on and not current_on:
                # Device turned ON
                current_on = True
                start_idx = idx
                energy_wh = 0.0
            elif is_on and current_on:
                # Continue ON session; accumulate energy using previous delta
                delta_h = float(row["delta_minutes"]) / 60.0
                energy_wh += power * delta_h
            elif not is_on and current_on:
                # Device turned OFF; close session
                end_idx = idx
                start_time = group.loc[start_idx, "timestamp"]
                end_time = row["timestamp"]
                duration_minutes = int(
                    (end_time - start_time).total_seconds() // 60
                ) or 1
                sessions.append(
                    DeviceUsageSession(
                        device=device,
                        usage_start=start_time,
                        usage_end=end_time,
                        duration_minutes=duration_minutes,
                        energy_kwh=round(energy_wh / 1000.0, 3),
                    )
                )
                current_on = False
                start_idx = None
                energy_wh = 0.0

        # If still ON at end of data, close at last timestamp
        if current_on and start_idx is not None:
            start_time = group.loc[start_idx, "timestamp"]
            end_time = group.iloc[-1]["timestamp"]
            duration_minutes = int((end_time - start_time).total_seconds() // 60) or 1

            # Approximate remaining energy using average power in the ON window
            on_window = group.loc[start_idx:]
            avg_power = float(on_window["power"].mean())
            delta_h = duration_minutes / 60.0
            energy_kwh = round((avg_power * delta_h) / 1000.0, 3)

            sessions.append(
                DeviceUsageSession(
                    device=device,
                    usage_start=start_time,
                    usage_end=end_time,
                    duration_minutes=duration_minutes,
                    energy_kwh=energy_kwh,
                )
            )

    return sessions


def analyze_daily_usage(
    df: pd.DataFrame, date: Optional[str] = None
) -> Tuple[List[DeviceUsageSession], DailyUsageSummary]:
    """
    Run full analysis for a single day:
    - Optionally select a specific date (YYYY-MM-DD); otherwise use the latest date.
    - Detect per-device usage sessions.
    - Aggregate hourly energy and compute summary statistics.
    """
    if df.empty:
        raise ValueError("No data available for analysis.")

    # Choose which calendar day to analyse
    all_dates = df["timestamp"].dt.date
    if date is not None:
        target_date = pd.to_datetime(date).date()
    else:
        target_date = all_dates.max()

    df_day = df[all_dates == target_date].copy()
    if df_day.empty:
        raise ValueError(f"No data available for date {target_date.isoformat()}.")

    sessions = _group_device_sessions(df_day)

    # Aggregate hourly energy usage (kWh) using simple trapezoidal approximation
    df_day["hour"] = df_day["timestamp"].dt.hour
    df_day["timestamp_shifted"] = df_day["timestamp"].shift(1)
    df_day["power_shifted"] = df_day["power"].shift(1)

    # Time delta between consecutive samples (minutes)
    df_day["delta_minutes"] = (
        (df_day["timestamp"] - df_day["timestamp_shifted"])
        .dt.total_seconds()
        .fillna(0)
        / 60.0
    )

    # Average power between two samples for better approximation
    avg_power = (df_day["power"] + df_day["power_shifted"].fillna(df_day["power"])) / 2.0
    df_day["energy_kwh"] = (avg_power * (df_day["delta_minutes"] / 60.0)) / 1000.0

    hourly_energy = (
        df_day.groupby("hour")["energy_kwh"].sum().reindex(range(24), fill_value=0.0)
    )

    total_energy_kwh = float(hourly_energy.sum())
    peak_hour = int(hourly_energy.idxmax())

    # "Most common usage time" interpreted as the hour with maximum number of ON samples
    on_mask = df_day["power"] > 0
    if on_mask.any():
        most_common_usage_hour = int(
            df_day.loc[on_mask, "hour"].value_counts().idxmax()
        )
    else:
        most_common_usage_hour = peak_hour

    date_str = target_date.isoformat()

    summary = DailyUsageSummary(
        date=date_str,
        total_energy_kwh=round(total_energy_kwh, 3),
        peak_hour=peak_hour,
        most_common_usage_hour=most_common_usage_hour,
        hourly_energy_kwh={int(h): round(float(v), 3) for h, v in hourly_energy.items()},
    )

    return sessions, summary


def predict_next_day_profile(summary: DailyUsageSummary) -> Dict[int, float]:
    """
    Use a simple LinearRegression model to extrapolate next-day hourly usage.

    This is intentionally lightweight for hackathon prototyping.
    """
    hours = np.array(list(summary.hourly_energy_kwh.keys())).reshape(-1, 1)
    values = np.array(list(summary.hourly_energy_kwh.values()))

    model = LinearRegression()
    model.fit(hours, values)

    next_day_hours = np.arange(24).reshape(-1, 1)
    predictions = model.predict(next_day_hours)
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    return {
        int(h): round(float(v), 3)
        for h, v in zip(next_day_hours.flatten(), predictions)
    }


__all__ = [
    "DeviceUsageSession",
    "DailyUsageSummary",
    "load_energy_data",
    "analyze_daily_usage",
    "predict_next_day_profile",
]