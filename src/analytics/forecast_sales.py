"""
Analytics: Forecast Sales (baseline models)

Input:  data/processed/sales_features.csv
Output:
  - images/forecast_plot.png
  - images/actual_vs_pred.png
  - prints model metrics (MAE, RMSE, MAPE)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "sales_features.csv"
IMAGES_DIR = REPO_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def train_test_split_time(df: pd.DataFrame, test_days: int = 60):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = len(df) - test_days
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def plot_actual_vs_pred(dates, actual, pred, title, out_path: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales Units")
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))   # fewer date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # readable format
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train, test = train_test_split_time(df, test_days=60)

    # -----------------------------
    # Model 1: Linear Regression (features -> sales)
    # -----------------------------
    feature_cols = [
        "day_of_week", "week_of_year", "month", "year",
        "promotion", "sales_lag_1", "sales_lag_7", "sales_roll_7", "sales_roll_28"
    ]

    X_train = train[feature_cols].values
    y_train = train["sales_units"].values
    X_test = test[feature_cols].values
    y_test = test["sales_units"].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_mape = mape(y_test, lr_pred)

    print("\n=== Linear Regression (feature-based) ===")
    print(f"MAE :  {lr_mae:,.2f}")
    print(f"RMSE:  {lr_rmse:,.2f}")
    print(f"MAPE:  {lr_mape:,.2f}%")

    plot_actual_vs_pred(
        test["date"], y_test, lr_pred,
        "Actual vs Predicted (Linear Regression)",
        IMAGES_DIR / "actual_vs_pred.png"
    )

    # -----------------------------
    # Model 2: ARIMA (univariate baseline)
    # -----------------------------
    y_train_ts = train.set_index("date")["sales_units"].asfreq("D")
    y_train_ts = y_train_ts.fillna(method="ffill")

    # Simple baseline order. Later we can tune this.
    arima = sm.tsa.ARIMA(y_train_ts, order=(2, 1, 2))
    arima_fit = arima.fit()

    steps = len(test)
    arima_forecast = arima_fit.forecast(steps=steps)
    arima_forecast = np.array(arima_forecast)

    arima_mae = mean_absolute_error(y_test, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(y_test, arima_forecast))
    arima_mape = mape(y_test, arima_forecast)

    print("\n=== ARIMA (2,1,2) baseline ===")
    print(f"MAE :  {arima_mae:,.2f}")
    print(f"RMSE:  {arima_rmse:,.2f}")
    print(f"MAPE:  {arima_mape:,.2f}%")

    # Forecast plot
    plt.figure()
    plt.plot(train["date"], train["sales_units"], label="Train")
    plt.plot(test["date"], y_test, label="Test (Actual)")
    plt.plot(test["date"], arima_forecast, label="Forecast (ARIMA)")
    plt.title("Sales Forecast (ARIMA baseline)")
    plt.xlabel("Date")
    plt.ylabel("Sales Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "forecast_plot.png", dpi=200)
    plt.close()

    print("\n✅ Saved images:")
    print(f"- {IMAGES_DIR / 'actual_vs_pred.png'}")
    print(f"- {IMAGES_DIR / 'forecast_plot.png'}")


if __name__ == "__main__":
    main()
