# Urban Mobility Intelligence

A local-first NYC taxi demand forecasting project. It cleans TLC trips, counts pickups by hour and zone, adds calendar and NOAA weather context, compares forecasts 1–6 hours ahead, flags unusual demand, and serves results through FastAPI and a Streamlit map dashboard.

## Requirements

- Python 3.11 or newer
- Internet access for the public TLC and NOAA downloads
- A few gigabytes of free disk space for source and processed data

## Install

Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the `py -3.11` launcher is unavailable, use your installed Python to create the virtual environment. Run all project commands from this folder.

## Prepare data

1. Download one monthly Yellow Taxi Parquet file from the [NYC TLC trip-record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and save it as `data/yellow_tripdata_2024-01.parquet` (or under `data/raw/` if you prefer).
2. Download the page’s **Taxi Zone Lookup Table (CSV)** to `data/raw/taxi_zone_lookup.csv`.
3. Inspect the source and build hourly demand:

```powershell
python -m src.audit_data --input data/yellow_tripdata_2024-01.parquet
python -m src.build_demand --input data/yellow_tripdata_2024-01.parquet --month 2024-01
python -m src.enrich_context
python -m src.add_weather
python -m src.download_zones
```

The outputs are saved under `data/processed/`. The weather step downloads NOAA Global Hourly observations for a Central Park station and joins them in New York local time. Weather is shared across zones; it is not zone-specific. The boundary download builds the polygons used by the map.

## Model and alerts

```powershell
python -m src.evaluate_baselines
python -m src.train_lightgbm
python -m src.train_horizons
python -m src.analyze_forecasts
python -m src.detect_anomalies
```

The six-horizon training command writes metrics to `reports/horizon_comparison.csv` and fitted models to `models/horizons/`. The error analysis writes zone/hour and feature-importance reports. The anomaly detector writes candidate alerts to `reports/anomaly_alerts.csv`; alerts need human review.

Run the automated pipeline check with:

```powershell
python -m pytest
```

## Run the API and dashboard

Start the API in PowerShell window 1:

```powershell
python -m uvicorn src.api:app --reload
```

Start the dashboard in PowerShell window 2:

```powershell
python -m streamlit run src/dashboard.py
```

Open the local Streamlit address (usually `http://localhost:8501`). The API documentation is at `http://127.0.0.1:8000/docs`. Keep both windows running while using the dashboard.

## Learn and present

- [Learning guide](docs/LEARNING_GUIDE.md) explains the core ideas in plain language.
- [Architecture](docs/ARCHITECTURE.md) shows how data, models, API, and dashboard connect.
- [Demo guide](docs/DEMO_GUIDE.md) includes a presentation walkthrough and likely technical questions.
- [Build plan](docs/PLAN.md) tracks the assignment requirements.
- [Data dictionary](docs/DATA_DICTIONARY.md) describes the processed hourly dataset fields.

## Important limits

The current demonstration uses January 2024 and evaluates the final week of that month. Its forecasts represent February 1, 2024, not current demand. Model quality may differ across seasons and years. The weather feature comes from one station, and anomaly flags are review candidates rather than confirmed incidents.

Raw downloaded datasets are kept local and excluded from Git. Follow the dataset source’s attribution and use terms when sharing the project.
