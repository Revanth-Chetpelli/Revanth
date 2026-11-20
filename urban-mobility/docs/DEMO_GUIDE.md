# Evaluation demo guide

## Before the demo

From the project folder, ensure dependencies, source data, processed weather data, taxi-zone boundaries, anomaly alerts, and six horizon models are available. If rebuilding the models, run:

```powershell
python -m src.train_horizons
```

Start two PowerShell windows in the project folder:

```powershell
python -m uvicorn src.api:app --reload
```

```powershell
python -m streamlit run src/dashboard.py
```

Open the local Streamlit URL (normally `http://localhost:8501`). Keep both terminal windows open during the demonstration.

## Suggested 5-minute walkthrough

1. **Problem:** Explain that planners need zone-level pickup forecasts and alerts for unusual changes.
2. **Data and pipeline:** Show that TLC trips are cleaned, checked for valid January timestamps, aggregated hourly by pickup zone, then enriched with calendar and NOAA weather context.
3. **Forecast map:** Change the forecast lead time from 1 to 6 hours. Point out high-demand zones and hover to show a zone name and predicted count.
4. **Zone history:** Select a busy zone such as JFK Airport or Times Square and explain the recent-count chart and forecast origin.
5. **Anomaly review:** Show the candidate alerts and explain that an alert means “review this unusual count,” not “fraud or disruption confirmed.”
6. **Evaluation:** Report MAE, RMSE, and WAPE and compare LightGBM with the same-hour-yesterday baseline. Emphasize the single-month, final-week test limitation.
7. **API:** Open `http://127.0.0.1:8000/docs` to show interactive endpoint documentation. Useful endpoints include `/health`, `/zones`, `/forecast/{zone_id}`, `/forecasts`, `/history/{zone_id}`, and `/anomalies`.

## Likely technical questions

**Why use a time-based split?** Forecasting uses past data to predict future data. A random split can leak later patterns into training and make results look unrealistically good.

**What does MAE mean?** Mean absolute error is the average absolute difference between predicted and actual pickup counts, in pickups.

**Why report WAPE?** It expresses total absolute error relative to total actual pickups, which helps compare zones with different demand volumes.

**Why keep a simple baseline?** It checks whether machine learning adds value beyond a transparent, easy-to-explain forecast.

**What does feature importance prove?** It shows which inputs the fitted model used most. It does not establish cause and effect.

**Are anomaly alerts confirmed incidents?** No. The statistical detector ranks unusual counts for human review; it does not verify their cause.

**What would improve the project next?** Train on several years or months, use rolling-origin evaluation, test weather impact across more periods, add prediction intervals, and connect a future-weather forecast source if operational forecasts are required.
