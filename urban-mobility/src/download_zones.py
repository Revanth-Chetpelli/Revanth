"""Download TLC taxi-zone boundaries and write a GeoJSON for the dashboard."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd


URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/raw/taxi_zones.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/taxi_zones.geojson"))
    args = parser.parse_args()
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.zip.exists():
        print("Downloading official TLC taxi-zone boundaries...")
        urlretrieve(URL, args.zip)
    unpacked = args.zip.parent / "taxi_zones_unpacked"
    with zipfile.ZipFile(args.zip) as archive:
        archive.extractall(unpacked)
    shapefile = unpacked / "taxi_zones" / "taxi_zones.shp"
    if not shapefile.exists():
        parser.error(f"Expected taxi-zone shapefile was not found in {args.zip}")
    zones = gpd.read_file(shapefile)
    if zones.crs is not None and zones.crs.to_epsg() != 4326:
        zones = zones.to_crs(epsg=4326)
    zones.to_file(args.output, driver="GeoJSON")
    print(f"Zones written: {len(zones):,}")
    print(f"Saved boundaries to: {args.output}")


if __name__ == "__main__":
    main()
