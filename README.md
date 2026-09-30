# Tidewise: AI Fish Price System

An end-to-end Coastal Karnataka fish price prediction system built around the architecture in the project brief:

```text
CSV / SQL history + weather
          ↓
Pandas + NumPy feature preparation
          ↓
Random Forest regression + chronological evaluation
          ↓
Flask REST API + SQLAlchemy database
          ↓
Responsive HTML / CSS / JavaScript dashboard
```

The dashboard is designed for Mangaluru, Udupi, Malpe, Karwar, Kundapura, and Honnavar. It uses fish species, location, temperature, rainfall, wind, humidity, wave conditions, available volume, buyers, quantity sold, demand, festival periods, season, month, and weekday behavior.

The dashboard also includes a profit calculator, best-selling location ranking, demand prediction, buy/sell decision support, live weather risk, supply forecast, price alerts, a market heatmap, browser location detection, and explainable feature importance. Landing volume supports scenarios up to 10,000 kg.

## Start the complete app

From PowerShell:

```powershell
cd "c:\AI_Fish_Price_ Detection"
C:/Users/Manvith/AppData/Local/Programs/Python/Python313/python.exe -m pip install -r requirements.txt
C:/Users/Manvith/AppData/Local/Programs/Python/Python313/python.exe generate_data.py
C:/Users/Manvith/AppData/Local/Programs/Python/Python313/python.exe train_model.py
C:/Users/Manvith/AppData/Local/Programs/Python/Python313/python.exe api.py
```

Open http://localhost:5000.

Do not run `app.py` directly. It is the earlier Streamlit prototype. The ready application is the Flask server in `api.py`.

## Project map

- `api.py`: Flask app, REST endpoints, weather integration, model loading.
- `database.py`: SQLAlchemy `FishPrice` table and serialization.
- `forecasting.py`: feature engineering, demo data, Random Forest training, prediction.
- `train_model.py`: trains and saves `models/coastal_karnataka_price_model.joblib`.
- `generate_data.py`: creates the multi-year demo CSV.
- `data/fish_prices.csv`: generated latest-five-years training data.
- `templates/index.html`: dashboard structure.
- `static/styles.css`: responsive reference-style dashboard design.
- `static/app.js`: API-connected interactions and SVG price chart.

## REST endpoints

```text
GET  /api/metadata
GET  /api/dashboard?location=Mangaluru&fish=Mackerel
POST /api/predict
POST /api/train
POST /api/seed
GET  /api/nearest-location?lat=12.91&lon=74.85
POST /api/insights
```

Example prediction request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5000/api/predict -ContentType 'application/json' -Body '{"location":"Mangaluru","fish":"Mackerel","demand_index":80,"catch_volume_kg":150}'
```

## Real historical CSV contract

Required columns:

```text
date,location,fish,price_per_kg,temperature_c,rainfall_mm,wind_speed_kmh
```

Recommended seller and buyer history columns:

```text
humidity_pct,wave_height_m,catch_volume_kg,buyers_count,quantity_sold_kg,demand_index,festival,season
```

Train on a real file:

```powershell
C:/Users/Manvith/AppData/Local/Programs/Python/Python313/python.exe train_model.py --csv "C:/path/to/real_coastal_karnataka_prices.csv"
```

Prices are expected in Indian rupees per kilogram. The demand index is normalized from 0 to 1000. `buyers_count` is the number of buyers for the fish/location/date, `quantity_sold_kg` is historical quantity sold, `festival` is 0 or 1, and `season` can be `Monsoon` or `Regular`. The current generated file is synthetic and is only for demonstrating the full pipeline. It must not be treated as today's actual auction price. Replace it with verified landing-centre, harbour, mandi, or market records for decision-grade output.

## Real-time decision inputs

The browser's **Use my location** button finds the nearest supported market from the user's coordinates. The server then requests current conditions from Open-Meteo for each market. If the weather service is temporarily unavailable, the app uses a seasonal fallback and labels the result as a planning estimate. The market heatmap compares predicted price, margin, demand, and risk across all six supported markets.

## Database

The default database is SQLite at `fish_prices.db`, created automatically and seeded from the CSV on first startup. To use MySQL, install a MySQL driver such as `pymysql` and change this setting in `api.py`:

```python
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://user:password@localhost/fish_prices"
```

Never commit production database credentials. Use environment variables before deployment.

## Model result from the included demo data

The included training run produced 81,216 training rows, 20,304 chronological holdout rows, ₹24.73/kg MAE, and R² 0.9817. These are validation results for generated demo data, not a guarantee of real-world performance.
