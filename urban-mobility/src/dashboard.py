"""Streamlit dashboard that consumes the local FastAPI service."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import geopandas as gpd
import json
import requests
import streamlit as st
from pathlib import Path


API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="NYC Mobility Intelligence", page_icon="🚕", layout="wide")
st.title("NYC Urban Mobility Intelligence")
st.caption("Hourly taxi pickup forecasts, recent zone demand, and candidate anomaly alerts")

try:
    zones = requests.get(f"{API_URL}/zones", timeout=10).json()
except requests.RequestException:
    st.error("The API is not running. Start it in another terminal with: python -m uvicorn src.api:app --reload")
    st.stop()

zone_options = {f"{z.get('zone_name', 'Unknown')} · {z.get('Borough', 'Unknown')} (ID {z['zone_id']})": z["zone_id"] for z in zones}
col_a, col_b = st.columns([3, 1])
selected_label = col_a.selectbox("Taxi zone", list(zone_options))
horizon = col_b.selectbox("Forecast lead time", [1, 2, 3, 4, 5, 6], format_func=lambda x: f"{x} hour{'s' if x > 1 else ''} ahead")
zone_id = zone_options[selected_label]

st.subheader("Forecast demand by taxi zone")
boundary_path = Path("data/processed/taxi_zones.geojson")
try:
    if not boundary_path.exists():
        st.info("Download the official TLC zone boundaries once with: python -m src.download_zones")
    else:
        bulk = requests.get(f"{API_URL}/forecasts", params={"horizon": horizon}, timeout=45)
        bulk.raise_for_status()
        forecast_frame = pd.DataFrame(bulk.json())
        boundary = gpd.read_file(boundary_path)
        id_column = next((col for col in ["LocationID", "location_id", "locationid"] if col in boundary.columns), None)
        if id_column is None:
            st.warning("The downloaded boundary file does not have a recognized zone ID field.")
        else:
            boundary["zone_id"] = pd.to_numeric(boundary[id_column], errors="coerce")
            map_data = boundary.merge(forecast_frame[["zone_id", "predicted_pickups"]], on="zone_id", how="left")
            demand = map_data["predicted_pickups"].fillna(0)
            map_data["predicted_pickups"] = demand
            map_geojson = json.loads(map_data.to_json())
            map_figure = px.choropleth_map(
                map_data, geojson=map_geojson, locations="LocationID",
                featureidkey="properties.LocationID", color="predicted_pickups",
                color_continuous_scale=[[0, "#2878be"], [0.25, "#00a18b"], [0.65, "#f5a623"], [1, "#cd313d"]],
                range_color=(0, max(200, float(demand.quantile(0.98)))), opacity=0.8,
                center={"lat": 40.72, "lon": -73.94}, zoom=9,
                hover_name="zone", hover_data={"borough": True, "predicted_pickups": ":.1f", "LocationID": False},
                labels={"predicted_pickups": "Forecast pickups", "borough": "Borough"},
            )
            map_figure.update_layout(
                map_style="carto-darkmatter", height=560,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                coloraxis_colorbar={"title": "Forecast pickups"},
            )
            st.plotly_chart(map_figure, width="stretch", key="zone_forecast_map_plotly", alt="NYC taxi-zone demand forecast map")
            st.caption(f"Forecast shading shows demand {horizon} hour(s) ahead. Blue indicates lower demand; red indicates higher demand. Hover over a zone for its name and predicted pickup count.")
except (requests.RequestException, ValueError, OSError) as exc:
    st.warning(f"The map could not be loaded: {exc}")

try:
    forecast = requests.get(f"{API_URL}/forecast/{zone_id}", params={"horizon": horizon}, timeout=30).json()
    hist_response = requests.get(f"{API_URL}/history/{zone_id}", params={"limit": 168}, timeout=15)
    hist_response.raise_for_status()
    history = pd.DataFrame(hist_response.json())
except (requests.RequestException, ValueError) as exc:
    st.error(f"Could not load forecast data: {exc}")
    st.stop()

metric_col, detail_col = st.columns([1, 2])
metric_col.metric("Predicted pickups", f"{forecast['predicted_pickups']:,.0f}", f"{horizon} hour(s) ahead")
detail_col.info(f"Forecast for {forecast['forecast_for']} · based on data through {forecast['forecast_origin']}. Weather forecast is not included; historical weather is used as context.")

history["hour"] = pd.to_datetime(history["hour"])
fig = px.line(history, x="hour", y="pickup_count", title="Recent hourly pickups", labels={"hour": "Hour", "pickup_count": "Pickups"})
fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=350)
st.plotly_chart(fig, width="stretch", alt="Recent hourly pickup counts for the selected taxi zone")

st.subheader("Candidate anomaly alerts")
try:
    alerts = requests.get(f"{API_URL}/anomalies", params={"limit": 100}, timeout=15).json()
    alert_frame = pd.DataFrame(alerts)
    if alert_frame.empty:
        st.info("No alerts are available. Generate them with python -m src.detect_anomalies.")
    else:
        st.dataframe(alert_frame, width="stretch", hide_index=True)
except requests.RequestException:
    st.warning("Anomaly alerts are unavailable. Run the anomaly detection step first.")

st.caption("Candidate alerts need human review. Forecast quality has only been evaluated on one month of data.")
