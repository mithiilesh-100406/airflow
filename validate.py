"""
validate.py
───────────
Task 3 of the Airflow DAG.
Runs business-rule checks on the cleaned DataFrame.
Raises ValueError on critical failures so Airflow marks the task as FAILED
and triggers the alert callback.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CLEANED_PATH = Path(os.getenv("CLEANED_DATA_PATH", "data/cleaned/sales_cleaned.csv"))


# ── Validation result container ────────────────────────────────────────────────

@dataclass
class ValidationReport:
    passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"✅ PASSED  ({len(self.passed)}): " + ", ".join(self.passed),
            f"⚠️  WARNINGS ({len(self.warnings)}): " + (", ".join(self.warnings) or "none"),
            f"❌ ERRORS  ({len(self.errors)}): " + (", ".join(self.errors) or "none"),
        ]
        return "\n".join(lines)


# ── Individual checks ──────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "order_id", "order_date", "customer_name",
    "region", "category", "sales", "quantity", "profit",
]


def check_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing columns: {missing}")
    else:
        report.passed.append("required_columns")


def check_row_count(df: pd.DataFrame, report: ValidationReport,
                    min_rows: int = 10) -> None:
    if len(df) < min_rows:
        report.errors.append(f"Too few rows: {len(df)} < {min_rows}")
    else:
        report.passed.append(f"row_count ({len(df):,})")


def check_no_negative_sales(df: pd.DataFrame, report: ValidationReport) -> None:
    if "sales" not in df.columns:
        return
    bad = (df["sales"] < 0).sum()
    if bad > 0:
        report.warnings.append(f"{bad} rows with negative sales")
    else:
        report.passed.append("no_negative_sales")


def check_date_range(df: pd.DataFrame, report: ValidationReport) -> None:
    if "order_date" not in df.columns:
        return
    oldest = df["order_date"].min()
    newest = df["order_date"].max()
    if pd.isna(oldest) or pd.isna(newest):
        report.errors.append("order_date contains all-null values")
        return
    if (newest - oldest).days > 365 * 10:
        report.warnings.append(f"Date range suspiciously wide: {oldest.date()} → {newest.date()}")
    else:
        report.passed.append(f"date_range ({oldest.date()} → {newest.date()})")


def check_discount_range(df: pd.DataFrame, report: ValidationReport) -> None:
    if "discount" not in df.columns:
        return
    out = df[(df["discount"] < 0) | (df["discount"] > 1)]
    if len(out) > 0:
        report.warnings.append(f"{len(out)} rows with discount outside [0,1]")
    else:
        report.passed.append("discount_range")


def check_null_threshold(df: pd.DataFrame, report: ValidationReport,
                          max_pct: float = 5.0) -> None:
    null_pct = df.isnull().mean() * 100
    bad_cols = null_pct[null_pct > max_pct]
    if not bad_cols.empty:
        report.warnings.append(
            f"High nulls (>{max_pct}%): " +
            ", ".join(f"{c}={v:.1f}%" for c, v in bad_cols.items())
        )
    else:
        report.passed.append("null_threshold")


CHECKS = [
    check_required_columns,
    check_row_count,
    check_no_negative_sales,
    check_date_range,
    check_discount_range,
    check_null_threshold,
]


# ── Main ───────────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> ValidationReport:
    logger.info("=== Starting validation ===")
    report = ValidationReport()

    for check in CHECKS:
        try:
            check(df, report)
        except Exception as e:
            report.errors.append(f"{check.__name__} raised exception: {e}")

    logger.info("\n" + report.summary())

    if not report.ok:
        raise ValueError(
            f"Validation FAILED with {len(report.errors)} error(s):\n" +
            "\n".join(report.errors)
        )

    return report


if __name__ == "__main__":
    from ingest import ingest_csv
    from clean import clean
    raw = ingest_csv()
    cleaned = clean(raw)
    validate(cleaned)
