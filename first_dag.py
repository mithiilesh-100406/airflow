from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mithilesh',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='first_dag',
    default_args=default_args,
    description='This is my first DAG',
    start_date=datetime(2026, 6, 17),  
    schedule='@daily'
) as dag:

    task1 = BashOperator(
        task_id='first_task',
        bash_command='echo "Hello World, this is my first Airflow task!"'
    )