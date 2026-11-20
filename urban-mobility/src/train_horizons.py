"""Train direct LightGBM forecasts for horizons 1 through 6 hours."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.train_lightgbm import make_panel


FEATURES = [
    "zone_id", "lag_1h", "lag_24h", "lag_168h", "rolling_mean_3h",
    "rolling_mean_24h", "hour_of_day", "day_of_week", "is_weekend",
    "temperature_c", "precipitation_mm", "is_us_federal_holiday",
    "temperature_missing", "precipitation_missing",
]


def evaluate_horizons(input_path: Path, output_path: Path, test_days: int, model_dir: Path) -> pd.DataFrame:
    panel = make_panel(input_path)
    group = panel.groupby("zone_id", sort=False)["actual"]
    panel["lag_1h"] = group.shift(1)
    panel["lag_24h"] = group.shift(24)
    panel["lag_168h"] = group.shift(168)
    panel["rolling_mean_3h"] = group.transform(lambda s: s.shift(1).rolling(3).mean())
    panel["rolling_mean_24h"] = group.transform(lambda s: s.shift(1).rolling(24).mean())
    panel["hour_of_day"] = panel["hour"].dt.hour
    panel["day_of_week"] = panel["hour"].dt.dayofweek
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")
    panel = panel.dropna(subset=FEATURES).copy()
    cutoff = panel["hour"].max() - pd.Timedelta(test_days, unit="D")
    results: list[dict[str, float | int | str]] = []

    for horizon in range(1, 7):
        rows = panel.copy()
        # Shift the future target back onto the hour when its forecast is made.
        rows["target"] = rows.groupby("zone_id", sort=False)["actual"].shift(-horizon)
        rows["target_hour"] = rows["hour"] + pd.Timedelta(horizon, unit="h")
        rows["baseline"] = rows.groupby("zone_id", sort=False)["actual"].shift(24 - horizon)
        rows = rows.dropna(subset=["target"])
        train = rows[rows["target_hour"] <= cutoff]
        test = rows[rows["target_hour"] > cutoff]
        if train.empty or test.empty:
            raise ValueError(f"Not enough data to evaluate the {horizon}-hour horizon.")

        evaluation_model = LGBMRegressor(
            objective="regression_l1", n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=42, verbosity=-1, n_jobs=-1,
        )
        evaluation_model.fit(train[FEATURES], train["target"])
        predicted = np.maximum(evaluation_model.predict(test[FEATURES]), 0)
        actual = test["target"].to_numpy()
        # For a target at issuance+h, its same hour yesterday is at issuance+h-24.
        # Thus the needed historical offset from issuance is 24-h hours.
        baseline = test["baseline"].to_numpy()
        total = actual.sum()
        for name, values in [("Same hour yesterday", baseline), ("LightGBM", predicted)]:
            results.append({
                "horizon_hours": horizon,
                "model": name,
                "test_rows": len(test),
                "MAE": float(mean_absolute_error(actual, values)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, values))),
                "WAPE_percent": float(np.abs(actual - values).sum() / total * 100) if total else float("nan"),
            })

        # Keep the held-out metrics honest, then refit the deployable model on
        # every labeled hour available so the final week is not discarded.
        deployment_model = LGBMRegressor(
            objective="regression_l1", n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=42, verbosity=-1, n_jobs=-1,
        )
        deployment_model.fit(rows[FEATURES], rows["target"])
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": deployment_model, "features": FEATURES, "horizon_hours": horizon}, model_dir / f"horizon_{horizon}.joblib")

    result = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand_weather.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/horizon_comparison.csv"))
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=Path("models/horizons"))
    args = parser.parse_args()
    try:
        result = evaluate_horizons(args.input, args.output, args.test_days, args.model_dir)
    except Exception as exc:
        parser.error(str(exc))
    print(result.to_string(index=False))
    print(f"\nSaved horizon results to: {args.output}")
    print(f"Saved six horizon models to: {args.model_dir}")


if __name__ == "__main__":
    main()
