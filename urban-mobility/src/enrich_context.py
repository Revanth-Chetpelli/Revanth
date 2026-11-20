"""Add readable NYC taxi-zone names and calendar context to hourly demand."""

from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import pandas as pd


def enrich(input_path: Path, lookup_path: Path, output_path: Path) -> pd.DataFrame:
    demand = pd.read_parquet(input_path)
    zones = pd.read_csv(lookup_path)
    needed = {"LocationID", "Borough", "Zone"}
    if not needed.issubset(zones.columns):
        raise ValueError(f"Zone lookup must contain columns: {sorted(needed)}")
    zones = zones[["LocationID", "Borough", "Zone"]].rename(columns={"LocationID": "zone_id", "Zone": "zone_name"})
    zones["zone_id"] = pd.to_numeric(zones["zone_id"], errors="coerce")
    demand["zone_id"] = pd.to_numeric(demand["zone_id"], errors="coerce")
    result = demand.merge(zones, on="zone_id", how="left", validate="many_to_one")
    # Keep observed trip counts even when a source zone ID is absent from the lookup.
    result["zone_name"] = result["zone_name"].fillna(result["zone_id"].map(lambda value: f"Unknown zone (ID {int(value)})"))
    result["Borough"] = result["Borough"].fillna("Unknown")
    result["hour"] = pd.to_datetime(result["hour"])
    date = result["hour"].dt.date
    us_holidays = holidays.US(years=sorted(result["hour"].dt.year.unique().tolist()))
    result["hour_of_day"] = result["hour"].dt.hour
    result["day_of_week"] = result["hour"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    result["is_us_federal_holiday"] = date.isin(us_holidays).astype("int8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand.parquet"))
    parser.add_argument("--lookup", type=Path, default=Path("data/raw/taxi_zone_lookup.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hourly_demand_context.parquet"))
    args = parser.parse_args()
    try:
        result = enrich(args.input, args.lookup, args.output)
    except Exception as exc:
        parser.error(str(exc))
    missing = int(result["zone_name"].isna().sum())
    print(f"Rows enriched: {len(result):,}")
    print(f"Rows without a matched zone name: {missing:,}")
    print(f"Federal holiday hours: {int(result['is_us_federal_holiday'].sum()):,}")
    print(f"Saved enriched data to: {args.output}")


if __name__ == "__main__":
    main()
