"""
Processing: Clean + Feature Engineering

Input:  data/raw/sales_data.csv
Output: data/processed/sales_features.csv
"""

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "sales_data.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "sales_features.csv"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Basic cleaning
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Feature engineering
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Lags (common forecasting features)
    df["sales_lag_1"] = df["sales_units"].shift(1)
    df["sales_lag_7"] = df["sales_units"].shift(7)

    # Rolling averages
    df["sales_roll_7"] = df["sales_units"].rolling(7).mean()
    df["sales_roll_28"] = df["sales_units"].rolling(28).mean()

    # Drop rows where lag/rolling features are NaN (first few days)
    df = df.dropna().reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"✅ Processed features written to: {OUT_PATH}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
