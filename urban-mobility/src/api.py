"""FastAPI service for forecasts, zone history, and anomaly review."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.train_horizons import FEATURES
from src.train_lightgbm import make_panel


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/hourly_demand_weather.parquet"
MODEL_DIR = ROOT / "models/horizons"
ALERT_PATH = ROOT / "reports/anomaly_alerts.csv"
app = FastAPI(title="Urban Mobility Demand API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": "urban-mobility"}


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Run python -m src.add_weather to prepare the demand data.")
    return pd.read_parquet(DATA_PATH)


def model_for(horizon: int):
    path = MODEL_DIR / f"horizon_{horizon}.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Train models with python -m src.train_horizons first.")
    return joblib.load(path)["model"]


def feature_panel(data: pd.DataFrame) -> pd.DataFrame:
    panel = make_panel(DATA_PATH)
    group = panel.groupby("zone_id", sort=False)["actual"]
    panel["lag_1h"] = group.shift(1)
    panel["lag_24h"] = group.shift(24)
    panel["lag_168h"] = group.shift(168)
    panel["rolling_mean_3h"] = group.transform(lambda s: s.shift(1).rolling(3).mean())
    panel["rolling_mean_24h"] = group.transform(lambda s: s.shift(1).rolling(24).mean())
    panel["hour_of_day"] = panel["hour"].dt.hour
    panel["day_of_week"] = panel["hour"].dt.dayofweek
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")
    return panel.dropna(subset=FEATURES)


@app.get("/zones")
def zones() -> list[dict[str, object]]:
    data = load_data()
    cols = [c for c in ["zone_id", "Borough", "zone_name"] if c in data]
    return data[cols].drop_duplicates("zone_id").sort_values("zone_id").to_dict("records")


@app.get("/forecast/{zone_id}")
def forecast(zone_id: int, horizon: int = Query(1, ge=1, le=6)) -> dict[str, object]:
    data = load_data()
    panel = feature_panel(data)
    rows = panel[panel["zone_id"] == zone_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Zone not found or not enough history.")
    row = rows.iloc[-1]
    model = model_for(horizon)
    # Transposing a mixed int/float pandas Series can coerce all columns to
    # object dtype; LightGBM accepts numeric arrays only.
    model_input = row[FEATURES].to_frame().T.astype("float64")
    prediction = max(0.0, float(model.predict(model_input)[0]))
    result = {"zone_id": zone_id, "horizon_hours": horizon,
              "forecast_for": str(row["hour"] + pd.Timedelta(hours=horizon)),
              "predicted_pickups": round(prediction, 2),
              "forecast_origin": str(row["hour"])}
    if "zone_name" in data:
        match = data[data["zone_id"] == zone_id].iloc[-1]
        result.update({"zone_name": match.get("zone_name"), "borough": match.get("Borough")})
    return result


@app.get("/forecasts")
def forecasts(horizon: int = Query(1, ge=1, le=6)) -> list[dict[str, object]]:
    data = load_data()
    panel = feature_panel(data)
    latest = panel.sort_values("hour").groupby("zone_id", as_index=False).tail(1).copy()
    model = model_for(horizon)
    latest["predicted_pickups"] = np.maximum(model.predict(latest[FEATURES].astype("float64")), 0)
    labels = data[["zone_id", "Borough", "zone_name"]].drop_duplicates("zone_id")
    latest = latest.merge(labels, on="zone_id", how="left")
    latest["forecast_for"] = latest["hour"] + pd.Timedelta(hours=horizon)
    return latest[["zone_id", "Borough", "zone_name", "forecast_for", "predicted_pickups"]].to_dict("records")


@app.get("/history/{zone_id}")
def history(zone_id: int, limit: int = Query(168, ge=1, le=744)) -> list[dict[str, object]]:
    data = load_data()
    rows = data[data["zone_id"] == zone_id].sort_values("hour").tail(limit)
    if rows.empty:
        raise HTTPException(status_code=404, detail="Zone not found.")
    return rows[["hour", "pickup_count"]].astype({"pickup_count": int}).to_dict("records")


@app.get("/anomalies")
def anomalies(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, object]]:
    if not ALERT_PATH.exists():
        raise HTTPException(status_code=503, detail="Run python -m src.detect_anomalies first.")
    alerts = pd.read_csv(ALERT_PATH).head(limit)
    data = load_data()
    labels = data[["zone_id", "Borough", "zone_name"]].drop_duplicates("zone_id")
    alerts = alerts.merge(labels, on="zone_id", how="left")
    return alerts.to_dict("records")
