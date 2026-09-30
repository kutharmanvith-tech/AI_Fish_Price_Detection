from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


LOCATIONS = {
    "Mangaluru": {"lat": 12.9141, "lon": 74.8560, "price_factor": 1.08},
    "Udupi": {"lat": 13.3409, "lon": 74.7421, "price_factor": 1.02},
    "Malpe": {"lat": 13.3490, "lon": 74.7031, "price_factor": 1.04},
    "Karwar": {"lat": 14.8167, "lon": 74.1297, "price_factor": 0.96},
    "Kundapura": {"lat": 13.6333, "lon": 74.6900, "price_factor": 0.98},
    "Honnavar": {"lat": 14.2800, "lon": 74.4500, "price_factor": 0.94},
}

FISH_BASE_PRICES = {
    "Mackerel": 280.0,
    "Sardine": 180.0,
    "Pomfret": 720.0,
    "Seer Fish": 850.0,
    "Tuna": 430.0,
    "Prawns": 640.0,
}

REQUIRED_COLUMNS = {
    "date",
    "location",
    "fish",
    "price_per_kg",
    "temperature_c",
    "rainfall_mm",
    "wind_speed_kmh",
}

FEATURE_COLUMNS = [
    "location",
    "fish",
    "temperature_c",
    "rainfall_mm",
    "wind_speed_kmh",
    "humidity_pct",
    "wave_height_m",
    "catch_volume_kg",
    "demand_index",
    "buyers_count",
    "quantity_sold_kg",
    "festival",
    "season",
    "month",
    "day_of_week",
    "is_monsoon",
    "is_weekend",
]


@dataclass(frozen=True)
class ModelReport:
    model: Pipeline
    mae: float
    r2: float
    train_rows: int
    test_rows: int


def _weather_for_day(current_date: pd.Timestamp, rng: np.random.Generator) -> dict[str, float]:
    month = current_date.month
    monsoon = month in (6, 7, 8, 9)
    return {
        "temperature_c": round((27.5 if monsoon else 29.3) + rng.normal(0, 1.4), 1),
        "rainfall_mm": round(max(0, (15 if monsoon else 2) + rng.gamma(1.3, 13 if monsoon else 2)), 1),
        "wind_speed_kmh": round(max(3, (17 if monsoon else 10) + rng.normal(0, 4)), 1),
        "humidity_pct": round(np.clip((84 if monsoon else 72) + rng.normal(0, 5), 45, 99), 1),
        "wave_height_m": round(max(0.2, (1.5 if monsoon else 0.8) + rng.normal(0, 0.25)), 2),
    }


def make_demo_data(start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    """Create a clearly labeled, deterministic dataset for local development.

    Real training should use the CSV uploader with observed mandi/landing-centre prices.
    """
    end_date = pd.Timestamp(end or date.today()) - pd.Timedelta(days=1)
    dates = pd.date_range(start=start, end=end_date, freq="D")
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for current_date in dates:
        weather = _weather_for_day(current_date, rng)
        monsoon = current_date.month in (6, 7, 8, 9)
        for location, location_profile in LOCATIONS.items():
            for fish, base_price in FISH_BASE_PRICES.items():
                seasonal = 1 + 0.12 * np.sin((current_date.dayofyear / 365) * 2 * np.pi)
                festival = int(current_date.month in (1, 4, 8, 10, 11) and current_date.day <= 10)
                buyers_count = max(20, int(rng.normal(180 if festival else 105, 24)))
                catch_volume = max(35, rng.normal(180 if not monsoon else 120, 32))
                quantity_sold = max(20, catch_volume * np.clip(0.64 + rng.normal(0, 0.08) + festival * 0.12, 0.3, 1))
                demand = np.clip(420 + buyers_count * 2.1 + quantity_sold * 0.7 + festival * 120 + (80 if current_date.dayofweek >= 5 else 0) + rng.normal(0, 55), 50, 1000)
                weather_pressure = 1 - (weather["wave_height_m"] * 0.025) + (weather["rainfall_mm"] * 0.001)
                price = base_price * location_profile["price_factor"] * seasonal
                price *= (1 + (demand - 500) / 2500) * weather_pressure * (1 + rng.normal(0, 0.055))
                rows.append(
                    {
                        "date": current_date,
                        "location": location,
                        "fish": fish,
                        "price_per_kg": round(max(price, 70), 2),
                        **weather,
                        "catch_volume_kg": round(catch_volume, 1),
                        "quantity_sold_kg": round(quantity_sold, 1),
                        "buyers_count": buyers_count,
                        "demand_index": round(demand, 1),
                        "festival": festival,
                        "season": "Monsoon" if monsoon else "Regular",
                    }
                )
    return pd.DataFrame(rows)


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["price_per_kg"] = pd.to_numeric(normalized["price_per_kg"], errors="coerce")
    for column in FEATURE_COLUMNS:
        if column not in normalized:
            normalized[column] = 0.0 if column not in {"location", "fish", "season"} else "Unknown"
    normalized = normalized.dropna(subset=["date", "price_per_kg", "location", "fish"])
    if normalized.empty:
        raise ValueError("No valid rows remain after parsing date and price_per_kg.")
    normalized["month"] = normalized["date"].dt.month
    normalized["day_of_week"] = normalized["date"].dt.dayofweek
    normalized["is_monsoon"] = normalized["month"].isin([6, 7, 8, 9]).astype(int)
    normalized["is_weekend"] = (normalized["day_of_week"] >= 5).astype(int)
    for column in FEATURE_COLUMNS:
        if column not in {"location", "fish", "season"}:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0)
    return normalized.sort_values("date").reset_index(drop=True)


def train_model(data: pd.DataFrame) -> ModelReport:
    prepared = prepare_data(data)
    split_index = max(int(len(prepared) * 0.8), 1)
    train = prepared.iloc[:split_index]
    test = prepared.iloc[split_index:]
    if test.empty:
        raise ValueError("At least two rows are required to create a chronological test split.")
    categorical = ["location", "fish", "season"]
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)]
    )
    pipeline = Pipeline(
        [
            ("features", preprocessor),
            ("model", RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(train[FEATURE_COLUMNS], train["price_per_kg"])
    predictions = pipeline.predict(test[FEATURE_COLUMNS])
    return ModelReport(
        model=pipeline,
        mae=float(mean_absolute_error(test["price_per_kg"], predictions)),
        r2=float(r2_score(test["price_per_kg"], predictions)),
        train_rows=len(train),
        test_rows=len(test),
    )


def predict_price(model: Pipeline, *, when: date, location: str, fish: str, weather: dict[str, float], catch_volume_kg: float, demand_index: float, buyers_count: float = 100, quantity_sold_kg: float | None = None, festival: int = 0, season: str | None = None) -> float:
    quantity_sold_kg = quantity_sold_kg if quantity_sold_kg is not None else catch_volume_kg * 0.7
    season = season or ("Monsoon" if when.month in (6, 7, 8, 9) else "Regular")
    prediction_row = pd.DataFrame(
        [
            {
                "location": location,
                "fish": fish,
                **weather,
                "catch_volume_kg": catch_volume_kg,
                "demand_index": demand_index,
                "buyers_count": buyers_count,
                "quantity_sold_kg": quantity_sold_kg,
                "festival": festival,
                "season": season,
                "month": when.month,
                "day_of_week": when.weekday(),
                "is_monsoon": int(when.month in (6, 7, 8, 9)),
                "is_weekend": int(when.weekday() >= 5),
            }
        ]
    )
    return float(max(model.predict(prediction_row[FEATURE_COLUMNS])[0], 0))
