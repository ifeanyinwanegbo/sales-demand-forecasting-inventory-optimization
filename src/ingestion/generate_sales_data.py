"""
Synthetic Sales Data Generator

Creates realistic daily sales data with:
- Trend
- Weekly seasonality
- Promotions
- Noise

Output: data/raw/sales_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"


def generate_sales_data(
    start_date="2022-01-01",
    end_date="2024-12-31",
    base_demand=200,
    trend_slope=0.05,
    promo_lift=0.25,
    noise_std=20,
):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    trend = trend_slope * np.arange(n)
    weekly_seasonality = 30 * np.sin(2 * np.pi * dates.dayofweek / 7)

    promotions = np.random.binomial(1, 0.15, size=n)
    promo_effect = promotions * base_demand * promo_lift

    noise = np.random.normal(0, noise_std, size=n)

    demand = base_demand + trend + weekly_seasonality + promo_effect + noise
    demand = np.maximum(demand, 0).round()

    df = pd.DataFrame(
        {
            "date": dates,
            "sales_units": demand,
            "promotion": promotions,
        }
    )

    return df


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_sales_data()
    output_path = RAW_DATA_DIR / "sales_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic sales data written to {output_path}")


if __name__ == "__main__":
    main()
