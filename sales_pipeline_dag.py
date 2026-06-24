"""
sales_pipeline_dag.py
─────────────────────
Place this file in ~/airflow/dags/ (or AIRFLOW_HOME/dags/).

Runs daily at 06:00 UTC.
DAG: ingest → clean → validate → embed
           ↘ (on any failure) → alert
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Make the pipeline package importable from the DAG ─────────────────────────
# Adjust this path to wherever you cloned the project
PROJECT_ROOT = Path("/opt/airflow/dags")
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingest   import ingest_csv
from pipeline.clean    import clean
from pipeline.validate import validate
from pipeline.embed    import embed_and_store


# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "email":            ["you@example.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}


# ── Task callables ────────────────────────────────────────────────────────────
# Each task uses XCom to pass the serialised DataFrame path to the next task
# (DataFrames are too large to push through XCom directly, so we use the
#  cleaned CSV path that clean.py writes to disk).

def task_ingest(**ctx):
    df = ingest_csv()
    ctx["ti"].xcom_push(key="row_count", value=len(df))


def task_clean(**ctx):
    df = ingest_csv()          # re-read from disk (stateless tasks)
    cleaned = clean(df)
    ctx["ti"].xcom_push(key="cleaned_rows", value=len(cleaned))


def task_validate(**ctx):
    import pandas as pd
    from dotenv import load_dotenv
    import os
    load_dotenv(str(PROJECT_ROOT / ".env"))
    cleaned_path = os.getenv("CLEANED_DATA_PATH", "data/cleaned/sales_cleaned.csv")
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    report = validate(df)
    ctx["ti"].xcom_push(key="validation_passed", value=len(report.passed))
    ctx["ti"].xcom_push(key="validation_warnings", value=len(report.warnings))


def task_embed(**ctx):
    import pandas as pd
    from dotenv import load_dotenv
    import os
    load_dotenv(str(PROJECT_ROOT / ".env"))
    cleaned_path = os.getenv("CLEANED_DATA_PATH", "data/cleaned/sales_cleaned.csv")
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    embed_and_store(df)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id          = "sales_csv_rag_pipeline",
    description     = "Daily: ingest → clean → validate → embed sales.csv into vector DB",
    default_args    = default_args,
    schedule        = "0 6 * * *",       # 06:00 UTC every day
    start_date      = datetime(2024, 1, 1),
    catchup         = False,
    tags            = ["sales", "rag", "llm"],
    max_active_runs = 1,
) as dag:

    ingest = PythonOperator(
        task_id         = "ingest",
        python_callable = task_ingest,
    )

    clean_task = PythonOperator(
        task_id         = "clean",
        python_callable = task_clean,
    )

    validate_task = PythonOperator(
        task_id         = "validate",
        python_callable = task_validate,
    )

    embed = PythonOperator(
        task_id         = "embed_to_vectorstore",
        python_callable = task_embed,
    )

    alert = EmailOperator(
        task_id      = "alert_on_failure",
        to           = ["you@example.com"],
        subject      = "⚠️ Sales RAG pipeline FAILED — {{ ds }}",
        html_content = """
            <h3>Pipeline failure on {{ ds }}</h3>
            <p>Task that failed: <b>{{ task_instance.task_id }}</b></p>
            <p>Check the Airflow UI for full logs.</p>
        """,
        trigger_rule = TriggerRule.ONE_FAILED,   # fires if ANY upstream task fails
    )

    # Chain: ingest → clean → validate → embed
    ingest >> clean_task >> validate_task >> embed

    # Alert fires on any failure (does NOT block the main chain)
    [ingest, clean_task, validate_task, embed] >> alert
