"""
clean.py
────────
Task 2 of the Airflow DAG.
Applies a sequence of cleaning transforms to the raw DataFrame and writes
the cleaned CSV to disk.

Expected columns (Kaggle Superstore / similar):
    order_id, order_date, ship_date, ship_mode, customer_id, customer_name,
    segment, country, city, state, postal_code, region, product_id,
    category, sub_category, product_name, sales, quantity, discount, profit
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CLEANED_PATH = Path(os.getenv("CLEANED_DATA_PATH", "data/cleaned/sales_cleaned.csv"))


# ── Individual cleaning steps (easy to unit-test) ─────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"drop_duplicates: removed {before - len(df)} rows")
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    logger.info(f"strip_whitespace: cleaned {len(str_cols)} text columns")
    return df


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    logger.info(f"standardise_column_names: {list(df.columns)}")
    return df


def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the correct types, coercing bad values to NaN."""
    date_cols = [c for c in df.columns if "date" in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    num_cols = ["sales", "quantity", "discount", "profit"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("enforce_types: dates and numerics cast")
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      - Numeric cols  → fill with 0 (quantity, discount) or median (sales, profit)
      - String cols   → fill with 'Unknown'
      - Date cols     → drop rows where order_date is missing
    """
    before = len(df)

    if "order_date" in df.columns:
        df = df.dropna(subset=["order_date"])
        logger.info(f"handle_nulls: dropped {before - len(df)} rows with null order_date")

    zero_fill = ["quantity", "discount"]
    for col in zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    median_fill = ["sales", "profit"]
    for col in median_fill:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].fillna("Unknown")

    logger.info(f"handle_nulls: remaining nulls = {df.isnull().sum().sum()}")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extra columns that are useful for LLM context."""
    if "order_date" in df.columns:
        df["order_year"]  = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month_name()
        df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)

    if "sales" in df.columns and "profit" in df.columns:
        df["profit_margin_pct"] = (
            (df["profit"] / df["sales"].replace(0, np.nan)) * 100
        ).round(2)

    logger.info("add_derived_columns: year, month, quarter, profit_margin_pct added")
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────────

CLEANING_STEPS = [
    standardise_column_names,
    drop_duplicates,
    strip_whitespace,
    enforce_types,
    handle_nulls,
    add_derived_columns,
]


def clean(df: pd.DataFrame, output_path: Path = CLEANED_PATH) -> pd.DataFrame:
    logger.info("=== Starting cleaning pipeline ===")
    for step in CLEANING_STEPS:
        df = step(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved → {output_path}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    from ingest import ingest_csv
    raw = ingest_csv()
    cleaned = clean(raw)
    print(cleaned.head())
