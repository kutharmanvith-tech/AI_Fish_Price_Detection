from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from forecasting import (
    FEATURE_COLUMNS,
    FISH_BASE_PRICES,
    LOCATIONS,
    make_demo_data,
    predict_price,
    prepare_data,
    train_model,
)


st.set_page_config(page_title="Tidewise | Coastal Karnataka Forecast", page_icon="🐟", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    return make_demo_data(start="2019-01-01")


@st.cache_resource
def train_cached(data: pd.DataFrame):
    return train_model(data)


@st.cache_data(ttl=900)
def fetch_current_weather(location: str) -> dict[str, float]:
    profile = LOCATIONS[location]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": profile["lat"],
            "longitude": profile["lon"],
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "Asia/Kolkata",
        },
        timeout=8,
    )
    response.raise_for_status()
    current = response.json()["current"]
    return {
        "temperature_c": float(current["temperature_2m"]),
        "rainfall_mm": float(current["precipitation"]),
        "wind_speed_kmh": float(current["wind_speed_10m"]),
        "humidity_pct": float(current["relative_humidity_2m"]),
        "wave_height_m": 1.5 if date.today().month in (6, 7, 8, 9) else 0.8,
    }


def fallback_weather() -> dict[str, float]:
    monsoon = date.today().month in (6, 7, 8, 9)
    return {
        "temperature_c": 28.0 if monsoon else 29.0,
        "rainfall_mm": 12.0 if monsoon else 2.0,
        "wind_speed_kmh": 16.0 if monsoon else 10.0,
        "humidity_pct": 84.0 if monsoon else 74.0,
        "wave_height_m": 1.5 if monsoon else 0.8,
    }


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink: #18353d; --muted: #6b7c80; --sea: #0e6571; --line: #dfe8e5; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: 0 !important; }
        .stApp { background: linear-gradient(135deg, #f8fbf8 0%, #eef6f3 52%, #fffaf2 100%); }
        [data-testid="stSidebar"] { background: #18353d; }
        [data-testid="stSidebar"] * { color: #eff7f3 !important; }
        [data-testid="stSidebar"] .stCaption { color: #b5cbc6 !important; }
        .brand { padding: 12px 0 24px; border-bottom: 1px solid rgba(255,255,255,.16); margin-bottom: 25px; }
        .brand-mark { display: inline-flex; width: 36px; height: 36px; align-items: center; justify-content: center; background: #f0b24c; color: #18353d; border-radius: 11px; font-size: 21px; margin-right: 9px; }
        .brand-name { font-family: 'Space Grotesk'; font-weight: 700; font-size: 22px; vertical-align: middle; }
        .eyebrow { color: var(--sea); font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 12px 0 5px; }
        .hero-title { font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.03; max-width: 750px; margin: 0; }
        .hero-copy { max-width: 720px; color: var(--muted); font-size: 16px; line-height: 1.55; margin-top: 13px; }
        .status-pill { display: inline-block; background: #dff0e6; color: #26704a; border-radius: 99px; padding: 6px 11px; font-size: 12px; font-weight: 700; }
        .metric-card { background: rgba(255,255,255,.84); border: 1px solid var(--line); border-radius: 14px; padding: 17px 19px; min-height: 108px; box-shadow: 0 9px 26px rgba(26,65,62,.055); }
        .metric-label { color: var(--muted); font-size: 13px; margin-bottom: 10px; }
        .metric-value { color: var(--ink); font-family: 'Space Grotesk'; font-size: 27px; font-weight: 700; }
        .metric-delta { color: #287c58; font-size: 12px; margin-top: 5px; }
        .section-title { font-size: 21px; margin: 27px 0 5px; }
        .section-copy { color: var(--muted); margin-top: 0; font-size: 14px; }
        .recommendation { background: #18353d; color: white; border-radius: 16px; padding: 21px; min-height: 188px; }
        .recommendation .kicker { color: #f0b24c; font-size: 11px; letter-spacing: .13em; font-weight: 700; text-transform: uppercase; }
        .recommendation h3 { color: white; font-size: 25px; margin: 10px 0 8px; }
        .recommendation p { color: #c9d9d5; line-height: 1.5; font-size: 14px; }
        .recommendation strong { color: #f0b24c; }
        .small-note { color: var(--muted); font-size: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">⌁</span><span class="brand-name">Tidewise</span></div>', unsafe_allow_html=True)
        st.caption("COASTAL KARNATAKA PRICE FORECAST")
        st.markdown("### Forecast inputs")
        forecast_date = st.date_input("Prediction date", value=date.today())
        location = st.selectbox("Coastal market", list(LOCATIONS), index=0)
        fish = st.selectbox("Fish species", list(FISH_BASE_PRICES), index=0)
        demand_index = st.slider("Expected demand", 30, 100, 75)
        catch_volume = st.number_input("Expected landing volume (kg)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
        st.divider()
        st.markdown("### Training data")
        source = st.radio("Data source", ["Built-in 2019–today demo", "Upload historical CSV"], label_visibility="collapsed")
        upload = st.file_uploader("Upload historical price data", type="csv", disabled=source.startswith("Built-in"))
        if source.startswith("Built-in"):
            st.caption("Demo data is synthetic. Upload observed landing-centre or market records for a real forecast.")
        else:
            st.caption("Required: date, location, fish, price_per_kg, temperature_c, rainfall_mm, wind_speed_kmh")

    if upload is not None and not source.startswith("Built-in"):
        try:
            data = prepare_data(pd.read_csv(upload))
            data_label = "Uploaded historical data"
        except (ValueError, KeyError) as error:
            st.error(f"Could not use this file: {error}")
            st.stop()
    else:
        data = load_demo_data()
        data_label = "Built-in synthetic demo data"

    try:
        report = train_cached(data)
    except ValueError as error:
        st.error(f"Training needs more usable historical data: {error}")
        st.stop()

    with st.sidebar:
        st.markdown('<span class="status-pill">● Model trained</span>', unsafe_allow_html=True)
        st.caption(f"{report.train_rows:,} training rows · {report.test_rows:,} holdout rows")
        st.download_button("Download demo CSV", data.to_csv(index=False), "coastal_karnataka_demo_prices.csv", mime="text/csv")

    weather = fallback_weather()
    weather_source = "Fallback estimate"
    try:
        weather = fetch_current_weather(location)
        weather_source = "Open-Meteo current weather"
    except (requests.RequestException, KeyError, TypeError, ValueError):
        pass

    prediction = predict_price(report.model, when=forecast_date, location=location, fish=fish, weather=weather, catch_volume_kg=catch_volume, demand_index=demand_index)
    lower = max(prediction - report.mae, 0)
    upper = prediction + report.mae

    st.markdown(f'<div class="eyebrow">Coastal Karnataka · {forecast_date.strftime("%d %B %Y")}</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Know the price before<br>the boat reaches shore.</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-copy">A trained price model for <strong>{location}</strong>, combining history, species, monsoon seasonality, landing volume, and current weather signals.</p>', unsafe_allow_html=True)

    cards = st.columns(4)
    with cards[0]:
        metric_card("Predicted price", f"₹{prediction:,.0f}/kg", f"Expected range ₹{lower:,.0f}–₹{upper:,.0f}")
    with cards[1]:
        metric_card("Market", location, weather_source)
    with cards[2]:
        metric_card("Model accuracy", f"₹{report.mae:,.0f} MAE", f"R² {report.r2:.2f} on holdout")
    with cards[3]:
        metric_card("Selected species", fish, f"Demand {demand_index}/100")

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<h2 class="section-title">Historical price signal</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-copy">Observed or generated price history for {location} and {fish}.</p>', unsafe_allow_html=True)
        chart_data = data[(data["location"] == location) & (data["fish"] == fish)].copy()
        chart = px.line(chart_data.tail(365), x="date", y="price_per_kg", labels={"price_per_kg": "Price / kg", "date": ""})
        chart.update_traces(line_color="#0e6571", line_width=2)
        chart.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        chart.update_xaxes(showgrid=False)
        chart.update_yaxes(showgrid=True, gridcolor="#e3ece8", tickprefix="₹", zeroline=False)
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<h2 class="section-title">Today’s selling decision</h2>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">The model’s price estimate with its tested error band.</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="recommendation"><div class="kicker">Forecast for {location}</div><h3>{fish} · ₹{prediction:,.0f}/kg</h3><p>Use <strong>₹{lower:,.0f}–₹{upper:,.0f}/kg</strong> as a practical planning range. The estimate is based on {data_label.lower()} and current weather inputs.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Weather signals used")
        weather_columns = st.columns(3)
        weather_columns[0].metric("Temperature", f"{weather['temperature_c']:.1f} °C")
        weather_columns[1].metric("Rainfall", f"{weather['rainfall_mm']:.1f} mm")
        weather_columns[2].metric("Wind", f"{weather['wind_speed_kmh']:.1f} km/h")

    st.markdown('<h2 class="section-title">Species outlook for this market</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Same weather and demand assumptions, ranked by predicted price for the selected location.</p>', unsafe_allow_html=True)
    ranking = []
    for species in FISH_BASE_PRICES:
        predicted = predict_price(report.model, when=forecast_date, location=location, fish=species, weather=weather, catch_volume_kg=catch_volume, demand_index=demand_index)
        ranking.append({"Species": species, "Predicted price / kg": predicted, "Planning range": f"₹{max(predicted - report.mae, 0):,.0f}–₹{predicted + report.mae:,.0f}"})
    ranking_data = pd.DataFrame(ranking).sort_values("Predicted price / kg", ascending=False)
    ranking_data["Predicted price / kg"] = ranking_data["Predicted price / kg"].map(lambda value: f"₹{value:,.0f}")
    st.dataframe(ranking_data, hide_index=True, use_container_width=True, height=260)
    st.markdown(f'<p class="small-note">Training source: {data_label}. Feature set includes {len(FEATURE_COLUMNS)} variables. A live estimate is not a guaranteed auction price; refresh with local observed prices as they become available.</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
