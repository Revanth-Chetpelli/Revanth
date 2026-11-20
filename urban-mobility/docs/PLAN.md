# Build plan

The assignment requires hourly demand forecasts 1–6 hours ahead by zone, anomaly detection, explanations, a FastAPI service, and a Streamlit map dashboard.

## Phases

1. **Data audit and setup:** choose a manageable NYC TLC period; inspect trip columns, timestamps, missing values, and zone coverage. Record data source and terms.
2. **Pipeline:** clean invalid trips, aggregate pickups/drop-offs by zone and 15/30/60 minute interval, join zone boundaries and weather, save partitioned Parquet.
3. **Features and baselines:** create lag, rolling, hour/day/holiday, weather, and neighboring-zone features. Compare seasonal-naive and moving-average forecasts.
4. **Models and validation:** compare LightGBM with baselines using chronological/rolling-origin backtests. Report MAE, RMSE, and WAPE per horizon and zone; prevent future data leakage.
5. **Anomalies and explanations:** robust z-score or Isolation Forest; review top alerts; explain feature importance without implying causation.
6. **Product layer:** FastAPI forecast/history/anomaly endpoints and Streamlit maps, filters, and time-series charts.
7. **Delivery:** README, data dictionary, architecture diagram, limitations, screenshots, and local demo instructions.

## Initial scope decision

Start with one recent, manageable TLC month, hourly pickup counts, and the available NYC taxi-zone lookup. Add weather after the basic pipeline works. This keeps the first version understandable while preserving the assignment's required design.
