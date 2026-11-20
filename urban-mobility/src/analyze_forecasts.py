"""Explain forecast errors by horizon, zone, and hour; summarize feature importance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.train_horizons import FEATURES
from src.train_lightgbm import make_panel


def analyze(input_path: Path, out_dir: Path, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = make_panel(input_path)
    grouped = panel.groupby("zone_id", sort=False)["actual"]
    panel["lag_1h"] = grouped.shift(1)
    panel["lag_24h"] = grouped.shift(24)
    panel["lag_168h"] = grouped.shift(168)
    panel["rolling_mean_3h"] = grouped.transform(lambda s: s.shift(1).rolling(3).mean())
    panel["rolling_mean_24h"] = grouped.transform(lambda s: s.shift(1).rolling(24).mean())
    panel["hour_of_day"] = panel["hour"].dt.hour
    panel["day_of_week"] = panel["hour"].dt.dayofweek
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")
    panel = panel.dropna(subset=FEATURES).copy()
    cutoff = panel["hour"].max() - pd.Timedelta(days=test_days)
    error_rows = []
    importance_rows = []

    for horizon in range(1, 7):
        frame = panel.copy()
        frame["target"] = frame.groupby("zone_id", sort=False)["actual"].shift(-horizon)
        frame["target_hour"] = frame["hour"] + pd.Timedelta(hours=horizon)
        frame = frame.dropna(subset=["target"])
        train = frame[frame["target_hour"] <= cutoff]
        test = frame[frame["target_hour"] > cutoff].copy()
        model = LGBMRegressor(objective="regression_l1", n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=42, verbosity=-1, n_jobs=-1)
        model.fit(train[FEATURES], train["target"])
        test["prediction"] = np.maximum(model.predict(test[FEATURES]), 0)
        test["absolute_error"] = np.abs(test["target"] - test["prediction"])
        test["hour_of_day_target"] = test["target_hour"].dt.hour
        test["horizon_hours"] = horizon
        error_rows.append(test[["horizon_hours", "zone_id", "hour_of_day_target", "target", "prediction", "absolute_error"]])
        gains = model.booster_.feature_importance(importance_type="gain")
        total_gain = gains.sum() or 1
        importance_rows.extend({"horizon_hours": horizon, "feature": feature,
                                "gain_importance_percent": float(gain / total_gain * 100)}
                               for feature, gain in zip(FEATURES, gains))

    errors = pd.concat(error_rows, ignore_index=True)
    zone = errors.groupby(["horizon_hours", "zone_id"], as_index=False).agg(
        test_rows=("absolute_error", "size"), MAE=("absolute_error", "mean"),
        actual_pickups=("target", "sum"), predicted_pickups=("prediction", "sum"))
    hour = errors.groupby(["horizon_hours", "hour_of_day_target"], as_index=False).agg(
        test_rows=("absolute_error", "size"), MAE=("absolute_error", "mean"))
    importance = pd.DataFrame(importance_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    zone.to_csv(out_dir / "error_by_zone.csv", index=False)
    hour.to_csv(out_dir / "error_by_hour.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    return zone, hour, importance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand_weather.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/error_analysis"))
    parser.add_argument("--test-days", type=int, default=7)
    args = parser.parse_args()
    if args.test_days < 1:
        parser.error("--test-days must be at least 1")
    try:
        zone, hour, importance = analyze(args.input, args.out_dir, args.test_days)
    except Exception as exc:
        parser.error(str(exc))
    print("Highest-error zones at 1-hour horizon:")
    print(zone[zone["horizon_hours"] == 1].nlargest(10, "MAE").to_string(index=False))
    print("\nError by target hour at 1-hour horizon:")
    print(hour[hour["horizon_hours"] == 1].sort_values("hour_of_day_target").to_string(index=False))
    print("\nTop feature importance across horizons:")
    print(importance.groupby("feature")["gain_importance_percent"].mean().nlargest(10).to_string())
    print(f"\nDetailed reports saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
