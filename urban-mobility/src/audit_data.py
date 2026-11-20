"""Read-only first-pass audit for an NYC TLC Parquet trip file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PICKUP_CANDIDATES = ("tpep_pickup_datetime", "lpep_pickup_datetime")
ZONE_CANDIDATES = ("PULocationID", "pulocationid")


def audit_file(path: Path) -> dict[str, object]:
    """Load a Parquet trip file and return a compact, beginner-friendly audit."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find input file: {path}")

    frame = pd.read_parquet(path)
    pickup_col = next((name for name in PICKUP_CANDIDATES if name in frame.columns), None)
    zone_col = next((name for name in ZONE_CANDIDATES if name in frame.columns), None)

    result: dict[str, object] = {
        "file": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing_by_column": frame.isna().sum().sort_values(ascending=False).head(12).to_dict(),
        "pickup_column": pickup_col,
        "pickup_zone_column": zone_col,
    }

    if pickup_col:
        timestamps = pd.to_datetime(frame[pickup_col], errors="coerce")
        valid = timestamps.dropna()
        result["invalid_pickup_timestamps"] = int(timestamps.isna().sum())
        result["pickup_date_min"] = str(valid.min()) if not valid.empty else "No valid timestamps"
        result["pickup_date_max"] = str(valid.max()) if not valid.empty else "No valid timestamps"
    else:
        result["warning"] = "No recognized pickup timestamp column; inspect the listed columns."

    if zone_col:
        result["unique_pickup_zones"] = int(frame[zone_col].nunique(dropna=True))
        result["missing_pickup_zones"] = int(frame[zone_col].isna().sum())

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to a TLC Parquet file")
    args = parser.parse_args()

    try:
        report = audit_file(args.input)
    except Exception as exc:
        parser.error(str(exc))

    print("\nDATA AUDIT")
    print("==========")
    for key, value in report.items():
        print(f"\n{key.replace('_', ' ').title()}:\n{value}")
    print("\nThis is an inspection only; the input file was not changed.")


if __name__ == "__main__":
    main()
