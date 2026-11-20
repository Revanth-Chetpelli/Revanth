"""Train a one-hour-ahead LightGBM model and compare it to yesterday's demand."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_panel(input_path: Path) -> pd.DataFrame:
    data = pd.read_parquet(input_path)
    data["hour"] = pd.to_datetime(data["hour"])
    context_columns = [name for name in ["temperature_c", "precipitation_mm", "is_us_federal_holiday"] if name in data.columns]
    context = data[["hour", *context_columns]].drop_duplicates("hour").copy() if context_columns else None
    hours = pd.date_range(data["hour"].min(), data["hour"].max(), freq="h")
    zones = np.sort(data["zone_id"].unique())
    index = pd.MultiIndex.from_product([hours, zones], names=["hour", "zone_id"])
    panel = (
        data.set_index(["hour", "zone_id"])["pickup_count"]
        .reindex(index, fill_value=0)
        .rename("actual")
        .reset_index()
        .sort_values(["zone_id", "hour"])
        .reset_index(drop=True)
    )
    if context is not None:
        panel = panel.merge(context, on="hour", how="left", validate="many_to_one")
        if "temperature_c" in panel:
            panel["temperature_missing"] = panel["temperature_c"].isna().astype("int8")
            panel["temperature_c"] = panel["temperature_c"].interpolate(limit_direction="both")
        if "precipitation_mm" in panel:
            panel["precipitation_missing"] = panel["precipitation_mm"].isna().astype("int8")
            panel["precipitation_mm"] = panel["precipitation_mm"].fillna(0)
        if "is_us_federal_holiday" in panel:
            panel["is_us_federal_holiday"] = panel["is_us_federal_holiday"].fillna(0).astype("int8")
    for name in ["temperature_c", "precipitation_mm", "is_us_federal_holiday", "temperature_missing", "precipitation_missing"]:
        if name not in panel:
            panel[name] = 0
    return panel


def train(input_path: Path, metrics_path: Path, model_path: Path, test_days: int) -> pd.DataFrame:
    panel = make_panel(input_path)
    by_zone = panel.groupby("zone_id", sort=False)["actual"]
    panel["lag_1h"] = by_zone.shift(1)
    panel["lag_24h"] = by_zone.shift(24)
    panel["lag_168h"] = by_zone.shift(168)
    panel["rolling_mean_3h"] = by_zone.transform(lambda s: s.shift(1).rolling(3).mean())
    panel["rolling_mean_24h"] = by_zone.transform(lambda s: s.shift(1).rolling(24).mean())
    panel["hour_of_day"] = panel["hour"].dt.hour
    panel["day_of_week"] = panel["hour"].dt.dayofweek
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")

    features = ["zone_id", "lag_1h", "lag_24h", "lag_168h", "rolling_mean_3h", "rolling_mean_24h", "hour_of_day", "day_of_week", "is_weekend"]
    panel = panel.dropna(subset=features).copy()
    cutoff = panel["hour"].max() - pd.Timedelta(days=test_days)
    train_rows = panel[panel["hour"] <= cutoff]
    test_rows = panel[panel["hour"] > cutoff].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError("Not enough dates for the requested time-based train/test split.")

    model = LGBMRegressor(
        objective="regression_l1", n_estimators=300, learning_rate=0.05,
        num_leaves=31, max_depth=-1, random_state=42, verbosity=-1,
        n_jobs=-1,
    )
    model.fit(train_rows[features], train_rows["actual"])
    test_rows["lightgbm_prediction"] = np.maximum(model.predict(test_rows[features]), 0)

    rows = []
    for label, prediction in [
        ("Same hour yesterday", "lag_24h"),
        ("LightGBM", "lightgbm_prediction"),
    ]:
        actual = test_rows["actual"].to_numpy()
        predicted = test_rows[prediction].to_numpy()
        total = actual.sum()
        rows.append({
            "model": label,
            "train_rows": len(train_rows),
            "test_hour_zone_rows": len(test_rows),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "WAPE_percent": float(np.abs(actual - predicted).sum() / total * 100) if total else float("nan"),
        })

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(metrics_path, index=False)
    import joblib
    joblib.dump({"model": model, "features": features}, model_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand.parquet"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/model_comparison.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/lightgbm_hourly.joblib"))
    parser.add_argument("--test-days", type=int, default=7)
    args = parser.parse_args()
    if args.test_days < 1:
        parser.error("--test-days must be at least 1")
    try:
        result = train(args.input, args.metrics, args.model, args.test_days)
    except Exception as exc:
        parser.error(str(exc))
    print(result.to_string(index=False))
    print(f"\nSaved metrics to: {args.metrics}")
    print(f"Saved model to: {args.model}")


if __name__ == "__main__":
    main()
