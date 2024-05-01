import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG(dag_id="Dag.02",
         start_date=datetime.datetime(2024, 1, 1),
         schedule="@daily"):
    EmptyOperator(task_id="task")