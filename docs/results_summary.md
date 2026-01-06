# Results Summary – Sales Demand Forecasting

This document summarizes the final outputs, model performance metrics, and generated artifacts from the sales demand forecasting and inventory optimization pipeline.

## Model Performance

### Linear Regression (Feature-Based)
| Metric | Value |
|------|------|
| MAE  | 21.90 |
| RMSE | 27.14 |
| MAPE | 8.31% |

### ARIMA (2,1,2) Baseline
| Metric | Value |
|------|------|
| MAE  | 35.64 |
| RMSE | 47.15 |
| MAPE | 12.71% |

## Interpretation

- The **Linear Regression model outperformed ARIMA** across all error metrics.
- Feature engineering (lags, rolling means, calendar variables, promotions) significantly improved forecast accuracy.
- ARIMA served as a strong univariate baseline but struggled with volatility and external signals.

## Generated Visualizations

- **Actual vs Predicted (Linear Regression)**  
  [`images/actual_vs_pred.png`](../images/actual_vs_pred.png)

- **Sales Forecast (ARIMA Baseline)**  
  [`images/forecast_plot.png`](../images/forecast_plot.png)

##  Dataset Overview

- **Time Range:** Multi-year daily sales data
- **Target Variable:** `sales_units`
- **Feature Engineering:**
  - Calendar features (day of week, week of year, month, year)
  - Promotion indicator
  - Lag features (1-day, 7-day)
  - Rolling averages (7-day, 28-day)

Processed dataset:
- [`data/processed/sales_features.csv`](../data/processed/sales_features.csv)

## 📌 Business Interpretation

- Feature-based forecasting reduced forecast error by **~35–40%** vs ARIMA.
- Improved forecast accuracy supports:
  - Lower safety stock requirements
  - Reduced overstock and stockouts
  - More informed procurement planning

This framework can be extended to:
- Gradient boosting / XGBoost
- Prophet
- Inventory optimization (EOQ, reorder points)

## 🧠 Takeaway

This project demonstrates how raw sales data can be transformed into
production-style forecasts using modular data engineering, analytics,
and statistical modeling best practices.
