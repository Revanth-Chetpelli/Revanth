"""Clean one monthly TLC file and aggregate pickup demand by hour and zone."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_hourly_demand(source: Path, output: Path, month: str) -> dict[str, int]:
    """Keep valid pickup rows within month, then count pickups per zone/hour."""
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find input file: {source}")
    try:
        start = pd.Timestamp(f"{month}-01")
    except ValueError as exc:
        raise ValueError("Month must use YYYY-MM format, for example 2024-01") from exc
    end = start + pd.offsets.MonthBegin(1)

    frame = pd.read_parquet(source, columns=["tpep_pickup_datetime", "PULocationID"])
    rows_read = len(frame)
    frame["tpep_pickup_datetime"] = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    frame["PULocationID"] = pd.to_numeric(frame["PULocationID"], errors="coerce")
    valid = frame[
        frame["tpep_pickup_datetime"].ge(start)
        & frame["tpep_pickup_datetime"].lt(end)
        & frame["PULocationID"].notna()
        & frame["PULocationID"].gt(0)
    ].copy()
    valid["zone_id"] = valid["PULocationID"].astype("int32")
    valid["hour"] = valid["tpep_pickup_datetime"].dt.floor("h")

    demand = (
        valid.groupby(["hour", "zone_id"], as_index=False)
        .size()
        .rename(columns={"size": "pickup_count"})
        .sort_values(["hour", "zone_id"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(output, index=False)
    return {
        "rows_read": rows_read,
        "rows_kept_in_month": len(valid),
        "rows_excluded": rows_read - len(valid),
        "hour_zone_rows": len(demand),
        "zones": int(demand["zone_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source TLC Parquet file")
    parser.add_argument("--month", required=True, help="Month represented by the file, YYYY-MM")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/hourly_demand.parquet"),
        help="Where to write the aggregated Parquet file",
    )
    args = parser.parse_args()
    try:
        report = build_hourly_demand(args.input, args.output, args.month)
    except Exception as exc:
        parser.error(str(exc))
    print("HOURLY DEMAND BUILD")
    print("===================")
    for name, value in report.items():
        print(f"{name.replace('_', ' ').title()}: {value:,}")
    print(f"\nSaved aggregated data to: {args.output}")


if __name__ == "__main__":
    main()
