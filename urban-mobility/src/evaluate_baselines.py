"""Compare simple one-hour-ahead forecasts on the final week of a month."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(input_path: Path, output_path: Path, test_days: int = 7) -> pd.DataFrame:
    data = pd.read_parquet(input_path)
    data["hour"] = pd.to_datetime(data["hour"])

    # Add zero-count rows for hours where a zone had no pickups.
    hours = pd.date_range(data["hour"].min(), data["hour"].max(), freq="h")
    zones = np.sort(data["zone_id"].unique())
    full_index = pd.MultiIndex.from_product([hours, zones], names=["hour", "zone_id"])
    panel = data.set_index(["hour", "zone_id"])["pickup_count"].reindex(full_index, fill_value=0).sort_index()
    panel = panel.rename("actual").reset_index()

    # Each forecast is made from information available before the target hour.
    panel["yesterday_same_hour"] = panel.groupby("zone_id")["actual"].shift(24)
    panel["recent_3_hour_average"] = (
        panel.groupby("zone_id")["actual"].transform(lambda values: values.shift(1).rolling(3).mean())
    )
    cutoff = hours.max() - pd.Timedelta(days=test_days)
    test = panel[(panel["hour"] > cutoff) & panel["yesterday_same_hour"].notna() & panel["recent_3_hour_average"].notna()].copy()

    rows: list[dict[str, float | str | int]] = []
    for name, prediction in [
        ("Same hour yesterday", "yesterday_same_hour"),
        ("Previous 3-hour average", "recent_3_hour_average"),
    ]:
        error = test["actual"].to_numpy() - test[prediction].to_numpy()
        denominator = test["actual"].sum()
        rows.append({
            "model": name,
            "test_hour_zone_rows": len(test),
            "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "WAPE_percent": float(np.abs(error).sum() / denominator * 100) if denominator else float("nan"),
        })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_metrics.csv"))
    parser.add_argument("--test-days", type=int, default=7)
    args = parser.parse_args()
    if args.test_days < 1:
        parser.error("--test-days must be at least 1")
    try:
        result = evaluate(args.input, args.output, args.test_days)
    except Exception as exc:
        parser.error(str(exc))
    print(result.to_string(index=False))
    print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
