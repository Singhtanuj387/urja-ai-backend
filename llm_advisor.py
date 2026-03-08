from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import APIError, AzureOpenAI

from .co2_calculator import EMISSION_FACTOR_KG_PER_KWH

load_dotenv()


def _get_openai_config() -> tuple[Optional[str], Optional[str], Optional[str], str]:
    """Read Azure OpenAI config from .env: api_key, base_url, api_version, model."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_version = os.getenv("OPENAI_API_VERSION") or None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return api_key, base_url, api_version, model


def _build_prompt(
    device: str,
    usage_window: str,
    grid_load: int,
    energy_kwh: float,
    co2_kg: float,
    recommended_time: str,
    potential_savings_kwh: float,
) -> str:
    """Create a concise, India-focused system prompt for the LLM."""
    potential_co2_saving = potential_savings_kwh * EMISSION_FACTOR_KG_PER_KWH
    return (
        "You are URJA-AI, an energy-efficiency assistant for Indian homes. "
        "Explain recommendations in simple, friendly language with one or two sentences. "
        "Focus on reducing electricity bills (₹) and CO₂ emissions, and mention grid stress when relevant.\n\n"
        f"Device: {device}\n"
        f"Current usage window: {usage_window}\n"
        f"Grid load during use: {grid_load}%\n"
        f"Energy used in that window: {energy_kwh:.2f} kWh\n"
        f"Estimated CO2 emission for this use: {co2_kg:.2f} kg\n"
        f"Recommended alternative time: {recommended_time}\n"
        f"Estimated potential energy saving: {potential_savings_kwh:.2f} kWh\n"
        f"Estimated potential CO2 saving: {potential_co2_saving:.2f} kg\n\n"
        "Generate a short recommendation explaining how shifting to the recommended time "
        "reduces both the electricity bill and CO₂ emissions, using Indian context "
        "(evening peak, late-night off-peak, etc.)."
    )


def _get_client() -> Optional[AzureOpenAI]:
    """
    Create an Azure OpenAI client using config from .env
    (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_API_VERSION).
    """
    api_key, base_url, api_version, _ = _get_openai_config()
    if not api_key or not base_url or not api_version:
        return None
    azure_endpoint = base_url.rstrip("/")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )


def generate_human_advice(
    *,
    device: str,
    usage_window: str,
    grid_load: int,
    energy_kwh: float,
    co2_kg: float,
    recommended_time: str,
    potential_savings_kwh: float,
) -> str:
    """
    Turn structured recommendation data into a human-friendly message.

    If OpenAI credentials are not configured or the API call fails, a simple
    rule-based fallback string is returned so the prototype still works.
    """
    client = _get_client()
    prompt = _build_prompt(
        device=device,
        usage_window=usage_window,
        grid_load=grid_load,
        energy_kwh=energy_kwh,
        co2_kg=co2_kg,
        recommended_time=recommended_time,
        potential_savings_kwh=potential_savings_kwh,
    )

    if client is None:
        # Fallback: deterministic text so the API is always usable.
        potential_co2_saving = potential_savings_kwh * EMISSION_FACTOR_KG_PER_KWH
        return (
            f"Your {device} usage from {usage_window} currently emits about {co2_kg:.2f} kg CO₂ "
            f"with grid load around {grid_load}%. Shifting it to around {recommended_time} can "
            f"save roughly {potential_savings_kwh:.2f} kWh and avoid about "
            f"{potential_co2_saving:.2f} kg CO₂ each time."
        )

    _, _, _, model_name = _get_openai_config()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful energy advisor."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=160,
            temperature=0.5,
        )
        message = response.choices[0].message.content or ""
        return message.strip()
    except APIError:
        potential_co2_saving = potential_savings_kwh * EMISSION_FACTOR_KG_PER_KWH
        return (
            f"Running your {device} around {recommended_time} instead of {usage_window} "
            f"is likely to reduce your bill and emissions, especially since this usage "
            f"releases about {co2_kg:.2f} kg CO₂ with grid load near {grid_load}%. "
            f"You could avoid roughly {potential_co2_saving:.2f} kg CO₂ each time."
        )


__all__ = ["generate_human_advice"]

