from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

from . import analyzer
from .co2_calculator import calculate_co2
from .grid_api import get_grid_load
from .llm_advisor import generate_human_advice
from .models import (
    AnalyzeResponse,
    DeviceUsage,
    DailyUsagePattern,
    HealthResponse,
    RecommendationDetails,
    RecommendationResponse,
)
from .recommendation_engine import suggest_optimal_time


SUPPORTED_DEVICES: List[str] = [
    "LED Bulb",
    "CFL Bulb",
    "Incandescent Bulb",
    "Ceiling Fan",
    "Table Fan",
    "Mobile Charger",
    "Laptop Charger",
    'LED TV (32")',
    'LED TV (50")',
    "Refrigerator (Compressor)",
    "Air Conditioner (1 Ton)",
    "Air Conditioner (1.5 Ton)",
    "Air Conditioner (2 Ton)",
    "Washing Machine",
    "Microwave Oven",
    "Electric Kettle",
    "Iron",
    "Water Heater (Geyser)",
    "Desktop Computer",
    "Mixer Grinder",
    "Vacuum Cleaner",
    "Hair Dryer",
    "Water Pump",
]


app = FastAPI(
    title="URJA-AI Backend",
    description="AI-powered smart energy advisor for Indian households.",
    version="0.1.0",
)


def _get_sample_data_path() -> str:
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "sample_data.json"
    csv_path = base_dir / "sample_data.csv"

    if json_path.exists():
        return str(json_path)
    if csv_path.exists():
        return str(csv_path)
    raise FileNotFoundError(f"No sample data found at {json_path} or {csv_path}")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", message="URJA-AI backend is running.")


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze_energy_usage(
    date: Optional[str] = Query(
        None, description="Date to analyze in YYYY-MM-DD format (defaults to latest)"
    ),
) -> AnalyzeResponse:
    """
    Analyze sample energy data and return device sessions + daily pattern.
    """
    try:
        data_path = _get_sample_data_path()
        df = analyzer.load_energy_data(data_path)
        sessions, summary = analyzer.analyze_daily_usage(df, date=date)
        predicted_profile = analyzer.predict_next_day_profile(summary)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    device_models: List[DeviceUsage] = [
        DeviceUsage(
            device=s.device,
            usage_start=s.usage_start.strftime("%H:%M"),
            usage_end=s.usage_end.strftime("%H:%M"),
            duration=s.duration_minutes,
            energy_kwh=s.energy_kwh,
        )
        for s in sessions
    ]

    summary_model = DailyUsagePattern(
        date=summary.date,
        total_energy_kwh=summary.total_energy_kwh,
        peak_hour=summary.peak_hour,
        most_common_usage_hour=summary.most_common_usage_hour,
        hourly_energy_kwh=summary.hourly_energy_kwh,
    )

    return AnalyzeResponse(
        devices=device_models,
        summary=summary_model,
        predicted_next_day_profile=predicted_profile,
        supported_devices=SUPPORTED_DEVICES,
    )


@app.get("/recommendation", response_model=RecommendationResponse)
def get_recommendation(
    date: Optional[str] = Query(
        None, description="Date to analyze in YYYY-MM-DD format (defaults to latest)"
    ),
) -> RecommendationResponse:
    """
    Run end-to-end flow:
    - Analyze sample data
    - Simulate grid load
    - Compute CO2 emissions
    - Generate LLM-based human-friendly recommendation
    """
    try:
        data_path = _get_sample_data_path()
        df = analyzer.load_energy_data(data_path)
        sessions, summary = analyzer.analyze_daily_usage(df, date=date)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    if not sessions:
        raise HTTPException(status_code=400, detail="No device usage sessions detected.")

    # Choose the highest-energy session as the primary candidate for optimization
    primary = max(sessions, key=lambda s: s.energy_kwh)
    usage_hour = primary.usage_start.hour

    grid_info = get_grid_load(usage_hour)
    grid_load = grid_info["grid_load"]

    reco = suggest_optimal_time(
        session=primary,
        grid_load_at_use=grid_load,
    )

    co2_kg = calculate_co2(primary.energy_kwh)
    potential_co2_saving_kg = calculate_co2(reco["energy_saving_kwh"])

    llm_message = generate_human_advice(
        device=primary.device,
        usage_window=primary.usage_time_str,
        grid_load=grid_load,
        energy_kwh=primary.energy_kwh,
        co2_kg=co2_kg,
        recommended_time=reco["recommended_time"],
        potential_savings_kwh=reco["energy_saving_kwh"],
    )

    usage_time_str = primary.usage_time_str

    details = RecommendationDetails(
        device=primary.device,
        usage_time=usage_time_str,
        grid_load=grid_load,
        recommended_time=reco["recommended_time"],
        energy_saving_kwh=reco["energy_saving_kwh"],
        co2_kg=co2_kg,
        potential_co2_saving_kg=potential_co2_saving_kg,
        co2_saving_suggestion=(
            f"If you follow this schedule, you can avoid roughly "
            f"{potential_co2_saving_kg:.2f} kg of CO₂ for this usage."
        ),
        raw_reason=reco["reason"],
        llm_message=llm_message,
    )

    return RecommendationResponse(
        device=primary.device,
        usage_time=usage_time_str,
        grid_load=grid_load,
        recommendation=llm_message,
        details=details,
    )


if __name__ == "__main__":
    # Convenience for local development: `python -m urja_ai.main`
    import uvicorn

    uvicorn.run(
        "urja_ai.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

