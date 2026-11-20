"""Download NOAA Global Hourly data and join weather to hourly demand."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


NOAA_STATION = "72505394728"  # Central Park, New York (USAF/WBAN)


def get_weather(year: int, csv_path: Path) -> pd.DataFrame:
    """Download one NOAA Global Hourly station-year CSV if it is not present."""
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", errors="ignore") as stream:
            first_line = stream.readline()
        if "DATE" not in first_line.upper() or first_line.lstrip().startswith("<"):
            # A previous request may have saved NOAA's HTML bucket page instead of a data file.
            csv_path.unlink()
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # NOAA moved the retired NCEI HTTPS archive to its NODD public S3 bucket.
        url = f"https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{NOAA_STATION}.csv"
        print(f"Downloading NOAA station data for {year}...")
        urlretrieve(url, csv_path)
    raw = pd.read_csv(csv_path, usecols=lambda c: c in {"DATE", "TMP", "AA1"}, dtype=str)
    if "DATE" not in raw.columns:
        raise ValueError("Downloaded file is not a NOAA Global Hourly CSV. Check the station URL and try again.")
    utc = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
    raw["hour"] = utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.floor("h")

    # NOAA TMP stores signed tenths of a degree Celsius before its quality flags.
    temp_tenths = pd.to_numeric(raw["TMP"].str.split(",").str[0], errors="coerce")
    temp_tenths = temp_tenths.mask(temp_tenths.abs() >= 9999)
    raw["temperature_c"] = temp_tenths / 10

    # AA1 fields: period, depth, amount, condition. Amount is tenths of millimeters.
    def precipitation_mm(value: object) -> float:
        if not isinstance(value, str) or not value:
            return float("nan")
        try:
            amount = value.split(",")[2]
            parsed = float(amount) if amount.strip() else float("nan")
            return parsed / 10 if abs(parsed) < 9999 else float("nan")
        except (IndexError, ValueError):
            return float("nan")

    raw["precipitation_mm"] = raw["AA1"].map(precipitation_mm)
    # Multiple observations can fall in an hour. Average temperature and retain
    # the largest reported precipitation amount to avoid summing overlapping reports.
    weather = raw.dropna(subset=["hour"]).groupby("hour", as_index=False).agg(
        temperature_c=("temperature_c", "mean"),
        precipitation_mm=("precipitation_mm", "max"),
    )
    return weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_demand_context.parquet"))
    parser.add_argument("--weather-csv", type=Path, default=Path("data/raw/noaa_central_park_2024.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hourly_demand_weather.parquet"))
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    try:
        demand = pd.read_parquet(args.input)
        weather = get_weather(args.year, args.weather_csv)
        demand["hour"] = pd.to_datetime(demand["hour"])
        result = demand.merge(weather, on="hour", how="left", validate="many_to_one")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
    except Exception as exc:
        parser.error(str(exc))
    missing = int(result["temperature_c"].isna().sum())
    print(f"Demand rows joined: {len(result):,}")
    print(f"Rows without a temperature observation: {missing:,}")
    print(f"Saved weather-enriched data to: {args.output}")


if __name__ == "__main__":
    main()
