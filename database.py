from __future__ import annotations

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class FishPrice(db.Model):
    __tablename__ = "fish_prices"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    location = db.Column(db.String(80), nullable=False, index=True)
    fish = db.Column(db.String(80), nullable=False, index=True)
    price_per_kg = db.Column(db.Float, nullable=False)
    temperature_c = db.Column(db.Float, nullable=False, default=0)
    rainfall_mm = db.Column(db.Float, nullable=False, default=0)
    wind_speed_kmh = db.Column(db.Float, nullable=False, default=0)
    humidity_pct = db.Column(db.Float, nullable=False, default=0)
    wave_height_m = db.Column(db.Float, nullable=False, default=0)
    catch_volume_kg = db.Column(db.Float, nullable=False, default=0)
    demand_index = db.Column(db.Float, nullable=False, default=0)
    buyers_count = db.Column(db.Float, nullable=False, default=0)
    quantity_sold_kg = db.Column(db.Float, nullable=False, default=0)
    festival = db.Column(db.Integer, nullable=False, default=0)
    season = db.Column(db.String(40), nullable=False, default="Regular")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "location": self.location,
            "fish": self.fish,
            "price_per_kg": self.price_per_kg,
            "temperature_c": self.temperature_c,
            "rainfall_mm": self.rainfall_mm,
            "wind_speed_kmh": self.wind_speed_kmh,
            "humidity_pct": self.humidity_pct,
            "wave_height_m": self.wave_height_m,
            "catch_volume_kg": self.catch_volume_kg,
            "demand_index": self.demand_index,
            "buyers_count": self.buyers_count,
            "quantity_sold_kg": self.quantity_sold_kg,
            "festival": self.festival,
            "season": self.season,
        }


class SupplyListing(db.Model):
    __tablename__ = "supply_listings"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    location = db.Column(db.String(80), nullable=False, index=True)
    fish = db.Column(db.String(80), nullable=False, index=True)
    available_volume_kg = db.Column(db.Float, nullable=False)
    expected_arrival = db.Column(db.Date, nullable=False)
    seller_name = db.Column(db.String(120), nullable=False, default="Anonymous seller")


class BuyerDemand(db.Model):
    __tablename__ = "buyer_demands"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    location = db.Column(db.String(80), nullable=False, index=True)
    fish = db.Column(db.String(80), nullable=False, index=True)
    buyers_count = db.Column(db.Integer, nullable=False)
    requested_volume_kg = db.Column(db.Float, nullable=False)
    needed_by = db.Column(db.Date, nullable=False)
    festival = db.Column(db.Boolean, nullable=False, default=False)