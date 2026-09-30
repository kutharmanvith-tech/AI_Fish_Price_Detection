from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sqlalchemy import inspect, text

from database import BuyerDemand, FishPrice, SupplyListing, db
from forecasting import FEATURE_COLUMNS, FISH_BASE_PRICES, LOCATIONS, predict_price, prepare_data
from train_model import train_from_csv


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "coastal_karnataka_price_model.joblib"
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fish_prices.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
CORS(app)


@lru_cache(maxsize=16)
def weather_for(location: str) -> dict[str, float]:
    profile = LOCATIONS[location]
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": profile["lat"], "longitude": profile["lon"], "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m", "timezone": "Asia/Kolkata"},
            timeout=6,
        )
        response.raise_for_status()
        current = response.json()["current"]
        return {"temperature_c": float(current["temperature_2m"]), "rainfall_mm": float(current["precipitation"]), "wind_speed_kmh": float(current["wind_speed_10m"]), "humidity_pct": float(current["relative_humidity_2m"]), "wave_height_m": 1.5 if date.today().month in (6, 7, 8, 9) else 0.8}
    except (requests.RequestException, KeyError, TypeError, ValueError):
        monsoon = date.today().month in (6, 7, 8, 9)
        return {"temperature_c": 28 if monsoon else 29, "rainfall_mm": 12 if monsoon else 2, "wind_speed_kmh": 16 if monsoon else 10, "humidity_pct": 84 if monsoon else 74, "wave_height_m": 1.5 if monsoon else 0.8}


def load_model():
    if not MODEL_PATH.exists():
        train_from_csv()
    return joblib.load(MODEL_PATH)


def model_explanation(model) -> list[dict[str, str | float]]:
    transformer = model.named_steps["features"]
    names = transformer.get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    ranked = sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:8]
    return [{"feature": name.replace("categorical__", "").replace("numeric__", "").replace("_", " ").title(), "importance": round(float(value) * 100, 2)} for name, value in ranked]


def weather_risk(weather: dict[str, float]) -> tuple[str, int]:
    risk = min(100, int(weather["rainfall_mm"] * 1.5 + weather["wind_speed_kmh"] * 1.5 + weather["wave_height_m"] * 18))
    label = "High" if risk >= 65 else "Moderate" if risk >= 35 else "Low"
    return label, risk


def economic_summary(predicted: float, quantity: float, buy_cost: float, operating_cost: float, demand: float) -> dict[str, float | str]:
    revenue = predicted * quantity
    total_cost = (buy_cost + operating_cost) * quantity
    profit = revenue - total_cost
    margin = (profit / revenue * 100) if revenue else 0
    action = "SELL" if margin >= 20 and demand >= 650 else "HOLD" if margin >= 8 else "BUY CAREFULLY"
    return {"quantity_kg": quantity, "buy_cost_per_kg": buy_cost, "operating_cost_per_kg": operating_cost, "revenue": round(revenue, 2), "total_cost": round(total_cost, 2), "profit": round(profit, 2), "margin_pct": round(margin, 2), "action": action}


def location_outlook(model, fish: str, when: date, quantity: float, buy_cost: float, operating_cost: float, demand: float, buyers_count: float, quantity_sold: float, festival: int) -> list[dict[str, object]]:
    outlook = []
    for location in LOCATIONS:
        location_weather = weather_for(location)
        predicted = predict_price(model, when=when, location=location, fish=fish, weather=location_weather, catch_volume_kg=quantity, demand_index=demand, buyers_count=buyers_count, quantity_sold_kg=quantity_sold, festival=festival)
        economics = economic_summary(predicted, quantity, buy_cost, operating_cost, demand)
        heat = min(100, max(0, int(demand / 10 * 0.55 + (predicted / max(FISH_BASE_PRICES[fish], 1)) * 35)))
        outlook.append({"location": location, "predicted_price": round(predicted, 2), "profit": economics["profit"], "margin_pct": economics["margin_pct"], "heat_score": heat, "demand": round(demand, 1), "action": economics["action"]})
    return sorted(outlook, key=lambda item: float(item["profit"]), reverse=True)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/metadata")
def metadata():
    return jsonify({"locations": list(LOCATIONS), "fish": list(FISH_BASE_PRICES), "features": FEATURE_COLUMNS})


@app.get("/api/nearest-location")
def nearest_location():
    try:
        latitude = float(request.args["lat"])
        longitude = float(request.args["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "lat and lon are required"}), 400
    nearest = min(LOCATIONS.items(), key=lambda item: (item[1]["lat"] - latitude) ** 2 + (item[1]["lon"] - longitude) ** 2)
    return jsonify({"location": nearest[0], "latitude": nearest[1]["lat"], "longitude": nearest[1]["lon"], "weather": weather_for(nearest[0])})


@app.get("/api/dashboard")
def dashboard():
    location = request.args.get("location", "Mangaluru")
    fish = request.args.get("fish", "Mackerel")
    rows = FishPrice.query.filter_by(location=location, fish=fish).order_by(FishPrice.date.desc()).limit(365).all()
    return jsonify({"history": [row.to_dict() for row in reversed(rows)], "weather": weather_for(location)})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    location = payload.get("location", "Mangaluru")
    fish = payload.get("fish", "Mackerel")
    if location not in LOCATIONS or fish not in FISH_BASE_PRICES:
        return jsonify({"error": "Unknown location or fish species"}), 400
    weather = payload.get("weather") or weather_for(location)
    model = load_model()
    quantity = min(max(float(payload.get("catch_volume_kg", 150)), 10), 10000)
    demand = min(max(float(payload.get("demand_index", 650)), 0), 1000)
    buyers_count = max(float(payload.get("buyers_count", demand / 5)), 0)
    quantity_sold = min(max(float(payload.get("quantity_sold_kg", quantity * 0.7)), 0), quantity)
    festival = int(payload.get("festival", 0))
    predicted = predict_price(model, when=date.fromisoformat(payload.get("date", date.today().isoformat())), location=location, fish=fish, weather=weather, catch_volume_kg=quantity, demand_index=demand, buyers_count=buyers_count, quantity_sold_kg=quantity_sold, festival=festival)
    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    buy_cost = max(float(payload.get("buy_cost_per_kg", predicted * 0.72)), 0)
    operating_cost = max(float(payload.get("operating_cost_per_kg", 35)), 0)
    risk_label, risk_score = weather_risk(weather)
    return jsonify({"location": location, "fish": fish, "predicted_price": round(predicted, 2), "lower": round(max(predicted - metrics.get("mae", 25), 0), 2), "upper": round(predicted + metrics.get("mae", 25), 2), "weather": weather, "weather_risk": {"label": risk_label, "score": risk_score}, "economics": economic_summary(predicted, quantity, buy_cost, operating_cost, demand), "metrics": metrics})


@app.post("/api/insights")
def insights():
    payload = request.get_json(silent=True) or {}
    location = payload.get("location", "Mangaluru")
    fish = payload.get("fish", "Mackerel")
    if location not in LOCATIONS or fish not in FISH_BASE_PRICES:
        return jsonify({"error": "Unknown location or fish species"}), 400
    when = date.fromisoformat(payload.get("date", date.today().isoformat()))
    quantity = min(max(float(payload.get("catch_volume_kg", 150)), 10), 10000)
    demand = min(max(float(payload.get("demand_index", 650)), 0), 1000)
    buyers_count = max(float(payload.get("buyers_count", demand / 5)), 0)
    quantity_sold = min(max(float(payload.get("quantity_sold_kg", quantity * 0.7)), 0), quantity)
    festival = int(payload.get("festival", 0))
    buy_cost = max(float(payload.get("buy_cost_per_kg", 0)), 0)
    operating_cost = max(float(payload.get("operating_cost_per_kg", 35)), 0)
    model = load_model()
    selected_weather = weather_for(location)
    selected_price = predict_price(model, when=when, location=location, fish=fish, weather=selected_weather, catch_volume_kg=quantity, demand_index=demand, buyers_count=buyers_count, quantity_sold_kg=quantity_sold, festival=festival)
    if buy_cost == 0:
        buy_cost = selected_price * 0.72
    economics = economic_summary(selected_price, quantity, buy_cost, operating_cost, demand)
    risk_label, risk_score = weather_risk(selected_weather)
    outlook = location_outlook(model, fish, when, quantity, buy_cost, operating_cost, demand, buyers_count, quantity_sold, festival)
    alerts = []
    if risk_score >= 65:
        alerts.append({"severity": "high", "title": "Weather risk", "message": "Wind, rain, or wave conditions may disrupt landings and transport."})
    if economics["margin_pct"] < 8:
        alerts.append({"severity": "high", "title": "Thin margin", "message": "Your costs leave little room for price movement or spoilage."})
    if demand >= 800:
        alerts.append({"severity": "positive", "title": "Demand surge", "message": "Demand is strong for this scenario. Consider selling sooner."})
    supply = max(0, round(quantity * (1 - risk_score / 180) * (0.65 + demand / 250), 1))
    return jsonify({"selected": {"location": location, "fish": fish, "price": round(selected_price, 2), "weather": selected_weather, "economics": economics, "weather_risk": {"label": risk_label, "score": risk_score}, "supply_prediction_kg": supply}, "best_location": outlook[0], "locations": outlook, "heatmap": [{"location": item["location"], "score": item["heat_score"], "price": item["predicted_price"], "profit": item["profit"]} for item in outlook], "alerts": alerts, "explanation": model_explanation(model)})


@app.post("/api/train")
def train():
    payload = request.get_json(silent=True) or {}
    csv_path = Path(payload.get("csv", ROOT / "data" / "fish_prices.csv"))
    try:
        return jsonify(train_from_csv(csv_path))
    except (OSError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/seed")
def seed_database():
    records = pd.read_csv(ROOT / "data" / "fish_prices.csv")
    prepared = prepare_data(records)
    db.session.query(FishPrice).delete()
    for row in prepared.to_dict("records"):
        db.session.add(FishPrice(**{column: row[column] for column in FishPrice.__table__.columns.keys() if column in row and column != "id" and column != "created_at"}))
    db.session.commit()
    return jsonify({"inserted": len(prepared)})


@app.post("/api/supply")
def create_supply():
    payload = request.get_json(silent=True) or {}
    try:
        listing = SupplyListing(date=date.fromisoformat(payload.get("date", date.today().isoformat())), location=payload["location"], fish=payload["fish"], available_volume_kg=min(max(float(payload["available_volume_kg"]), 0), 10000), expected_arrival=date.fromisoformat(payload.get("expected_arrival", date.today().isoformat())), seller_name=payload.get("seller_name", "Anonymous seller"))
        if listing.location not in LOCATIONS or listing.fish not in FISH_BASE_PRICES:
            raise ValueError("Unknown location or fish species")
        db.session.add(listing)
        db.session.commit()
        return jsonify({"id": listing.id, "message": "Supply listing saved"}), 201
    except (KeyError, TypeError, ValueError) as error:
        db.session.rollback()
        return jsonify({"error": str(error)}), 400


@app.post("/api/demand")
def create_demand():
    payload = request.get_json(silent=True) or {}
    try:
        demand = BuyerDemand(date=date.fromisoformat(payload.get("date", date.today().isoformat())), location=payload["location"], fish=payload["fish"], buyers_count=min(max(int(payload["buyers_count"]), 0), 10000), requested_volume_kg=min(max(float(payload["requested_volume_kg"]), 0), 10000), needed_by=date.fromisoformat(payload.get("needed_by", date.today().isoformat())), festival=bool(payload.get("festival", False)))
        if demand.location not in LOCATIONS or demand.fish not in FISH_BASE_PRICES:
            raise ValueError("Unknown location or fish species")
        db.session.add(demand)
        db.session.commit()
        return jsonify({"id": demand.id, "message": "Buyer demand saved"}), 201
    except (KeyError, TypeError, ValueError) as error:
        db.session.rollback()
        return jsonify({"error": str(error)}), 400


def migrate_database() -> None:
    """Add newly introduced feature columns to an existing SQLite database."""
    inspector = inspect(db.engine)
    existing = {column["name"] for column in inspector.get_columns("fish_prices")}
    additions = {
        "buyers_count": "FLOAT NOT NULL DEFAULT 0",
        "quantity_sold_kg": "FLOAT NOT NULL DEFAULT 0",
        "festival": "INTEGER NOT NULL DEFAULT 0",
        "season": "VARCHAR(40) NOT NULL DEFAULT 'Regular'",
    }
    for name, definition in additions.items():
        if name not in existing:
            db.session.execute(text(f"ALTER TABLE fish_prices ADD COLUMN {name} {definition}"))
    db.session.commit()


with app.app_context():
    db.create_all()
    migrate_database()
    if not FishPrice.query.first():
        seed_database()


if __name__ == "__main__":
    app.run(debug=True, port=5000)