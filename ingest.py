"""
ingest.py
─────────
Task 1 of the Airflow DAG.
Loads the raw sales.csv, archives a timestamped backup, and returns the
DataFrame for the next task.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/sales.csv"))
ARCHIVE_DIR = Path("data/archive")


def ingest_csv(filepath: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load sales.csv → DataFrame.
    Archives a timestamped copy so every daily run is preserved.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Sales CSV not found at {filepath}")

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, low_memory=False, encoding="latin-1")
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Archive raw file ──────────────────────────────────────────────────────
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"sales_{stamp}.csv"
    shutil.copy(filepath, archive_path)
    logger.info(f"Archived raw file → {archive_path}")

    return df


if __name__ == "__main__":
    df = ingest_csv()
    print(df.head())
    print(df.dtypes)
