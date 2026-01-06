"""
Sales Demand Forecasting & Inventory Optimization
Pipeline Orchestrator

This script will run the project end-to-end:
1) Generate/ingest data
2) Clean + transform
3) Train forecasting models
4) Evaluate accuracy
5) Produce charts + outputs for the README

We will fill each module step-by-step.
"""

from __future__ import annotations

from pathlib import Path


# Project paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
IMAGES_DIR = REPO_ROOT / "images"


def ensure_folders() -> None:
    """Create required folders if they don't exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_folders()
    print("✅ Folder structure ready.")
    print(f"Raw data folder:       {DATA_RAW_DIR}")
    print(f"Processed data folder: {DATA_PROCESSED_DIR}")
    print(f"Images folder:         {IMAGES_DIR}")

    # Next steps (we’ll implement these in the next files):
    print("\nNext: we will generate synthetic sales data into data/raw/")


if __name__ == "__main__":
    main()
