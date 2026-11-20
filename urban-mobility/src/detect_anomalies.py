"""Flag unusual zone demand using historical same-hour-of-week robust scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_lightgbm import make_panel


def detect(input_path: Path, output_path: Path, threshold: float, min_history: int) -> pd.DataFrame:
    panel = make_panel(input_path)
    panel["hour_of_week"] = panel["hour"].dt.dayofweek * 24 + panel["hour"].dt.hour
    group = panel.groupby(["zone_id", "hour_of_week"], sort=False)["actual"]
    panel["historical_median"] = group.transform(lambda s: s.shift(1).expanding(min_periods=min_history).median())
    panel["historical_mad"] = group.transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).apply(
            lambda values: np.median(np.abs(values - np.median(values))), raw=True
        )
    )
    # Scale MAD to estimate standard deviation. For count data with MAD=0,
    # use a Poisson-style sqrt(count) scale so constant/sparse histories remain scoreable.
    mad_scale = 1.4826 * panel["historical_mad"]
    count_scale = np.sqrt(panel["historical_median"].clip(lower=0) + 1)
    scale = pd.concat([mad_scale, count_scale, pd.Series(1.0, index=panel.index)], axis=1).max(axis=1)
    panel["robust_z_score"] = (panel["actual"] - panel["historical_median"]) / scale
    alerts = panel[panel["robust_z_score"].abs() >= threshold].copy()
    alerts["direction"] = np.where(alerts["robust_z_score"] > 0, "spike", "drop")
    alerts = alerts[["hour", "zone_id", "actual", "historical_median", "robust_z_score", "direction"]]
    alerts = alerts.sort_values("robust_z_score", key=lambda col: col.abs(), ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(output_path, index=False)
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly_alerts.csv"))
    parser.add_argument("--threshold", type=float, default=3.5, help="Absolute robust z-score threshold")
    parser.add_argument("--min-history", type=int, default=2, help="Prior matching weekdays needed")
    args = parser.parse_args()
    if args.threshold <= 0 or args.min_history < 1:
        parser.error("threshold must be positive and min-history must be at least 1")
    try:
        alerts = detect(args.input, args.output, args.threshold, args.min_history)
    except Exception as exc:
        parser.error(str(exc))
    print(f"Candidate alerts: {len(alerts):,}")
    if alerts.empty:
        print("No demand values crossed the chosen robust z-score threshold.")
    else:
        print(alerts.head(20).to_string(index=False))
    print("\nAlerts are candidates for human review, not confirmed disruptions.")
    print(f"Saved alert list to: {args.output}")


if __name__ == "__main__":
    main()
