from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

from .analyzer import DeviceUsageSession


def _format_hour(hour: int) -> str:
    """Format an integer hour (0–23) as HH:MM."""
    hour = max(0, min(23, hour))
    return f"{hour:02d}:00"


def suggest_optimal_time(
    session: DeviceUsageSession,
    grid_load_at_use: int,
    baseline_grid_load: Optional[int] = None,
) -> Dict[str, object]:
    """
    Simple rule-based recommendation:

    - If grid_load_at_use > 80%, suggest shifting usage by +2–3 hours.
    - Otherwise, keep the same time and highlight efficient usage.
    """
    use_start_hour = session.usage_start.hour
    recommended_hour = use_start_hour
    reason: str
    savings_kwh: float = 0.0

    if grid_load_at_use > 80:
        shift_hours = 3 if 18 <= use_start_hour <= 22 else 2
        recommended_hour = (use_start_hour + shift_hours) % 24
        baseline = baseline_grid_load if baseline_grid_load is not None else max(
            30, grid_load_at_use - 20
        )
        relative_drop = max(grid_load_at_use - baseline, 5) / max(
            grid_load_at_use, 1
        )
        savings_kwh = round(session.energy_kwh * relative_drop * 0.5, 3)
        reason = (
            f"Grid load is high at {use_start_hour:02d}:00 ({grid_load_at_use}%). "
            f"Shifting usage to {recommended_hour:02d}:00 avoids peak demand."
        )
    else:
        reason = (
            f"Grid load is moderate at {use_start_hour:02d}:00 ({grid_load_at_use}%). "
            "Current usage time is already reasonably efficient."
        )

    recommended_time_str = _format_hour(recommended_hour)
    usage_time_str = session.usage_time_str

    return {
        "device": session.device,
        "current_usage_window": usage_time_str,
        "recommended_time": recommended_time_str,
        "reason": reason,
        "energy_saving_kwh": savings_kwh,
    }


__all__ = ["suggest_optimal_time"]

