import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG(dag_id="Dag.03",
         start_date=pendulum.datetime(2023, 1, 1, tz='America/Bogota'),
         schedule="*/10 * * * *",
         tags=['dev', 'test', 'IA']):
     print('hola 03')
     EmptyOperator(task_id="task")