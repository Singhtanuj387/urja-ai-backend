from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DeviceUsage(BaseModel):
    device: str
    usage_start: str = Field(..., description="Start time in HH:MM")
    usage_end: str = Field(..., description="End time in HH:MM")
    duration: int = Field(..., description="Duration in minutes")
    energy_kwh: float
    co2_kg: float = Field(..., description="Estimated CO₂ emissions for this usage (kg)")


class DailyUsagePattern(BaseModel):
    date: str
    total_energy_kwh: float
    total_co2_kg: float = Field(
        ..., description="Estimated total CO₂ emissions for the day (kg)"
    )
    peak_hour: int
    most_common_usage_hour: int
    hourly_energy_kwh: Dict[int, float]


class AnalyzeResponse(BaseModel):
    devices: List[DeviceUsage]
    summary: DailyUsagePattern
    predicted_next_day_profile: Dict[int, float]
    supported_devices: List[str]


class RecommendationDetails(BaseModel):
    device: str
    usage_time: str
    grid_load: int
    recommended_time: str
    energy_saving_kwh: float
    co2_kg: float
    potential_co2_saving_kg: float
    co2_saving_suggestion: str
    raw_reason: str
    llm_message: str


class RecommendationResponse(BaseModel):
    device: str
    usage_time: str
    grid_load: int
    recommendation: str
    details: RecommendationDetails


class HealthResponse(BaseModel):
    status: str
    message: Optional[str] = None


__all__ = [
    "DeviceUsage",
    "DailyUsagePattern",
    "AnalyzeResponse",
    "RecommendationDetails",
    "RecommendationResponse",
    "HealthResponse",
]

