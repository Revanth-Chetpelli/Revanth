# Learning guide

## What are we building?

At each hour, count taxi pickups in each NYC zone. Use earlier counts plus calendar and weather context to estimate demand 1–6 hours into the future. Separately flag counts that look unusually high or low.

## Key terms

- **Zone:** a named geographic area used by the taxi data.
- **Feature:** information given to a model, such as hour of day or demand one hour ago.
- **Target:** the value we want to predict, here future pickup count.
- **Baseline:** a simple reference forecast. A useful advanced model must beat it.
- **Time-based validation:** train on earlier dates and evaluate on later dates, like real forecasting.
- **Data leakage:** accidentally letting future information help predict the past.
- **API:** a defined way for another program to request forecasts.
- **Dashboard:** the visual interface for exploring forecasts and alerts.

We will explain each file as it is added and build one working layer at a time.
