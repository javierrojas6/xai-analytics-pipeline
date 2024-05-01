import logging
import os
import sys

import pendulum
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor


sys.path.append(os.path.join(os.getcwd(), './src'))
from app.core.file.folder import trim_directory
from app.core.file.folder import prepare_directory
from app.core.database.backup import build_backup_bash_command


with DAG(dag_id="Abigail.backup.01",
         start_date=pendulum.datetime(2024, 1, 1, tz='America/Bogota'),
         schedule="*/10 * * * *",
         #  schedule="@daily",
         tags=['abigail', 'backup'],
         orientation = 'TB',
         max_active_tasks = 1,
         max_active_runs = 1) as dag:

    backup_directory = 'backups/daily'
    label = 'daily'
    preserve_files = 24

    logging.info(f'starting {label}')

    load_dotenv()

    settings = {
        "database": os.environ.get('TARGET_DB_DATABASE'),
        "-h": os.environ.get('TARGET_DB_HOST'),
        "-p": os.environ.get('TARGET_DB_PORT'),
        "-U": os.environ.get('TARGET_DB_USER'),
        "-T": "datalake.file",
        "--inserts": "",
        "-a": "",
    }

    env = {'PGPASSWORD': os.environ.get('TARGET_DB_PASSWORD')}

    prepare_directory_op = PythonOperator(task_id=f'prepare-{label}-backup-directory',
                                          python_callable=prepare_directory,
                                          op_kwargs={'path': backup_directory},
                                          dag=dag,
                                          owner='abigail',
                                          doc='prepare destiny directory where all backups will stored to')

    bash_command, filename = build_backup_bash_command(settings=settings,
                                                       directory=os.path.join(os.getcwd(), backup_directory))

    backup_command_op = BashOperator(task_id=f"postgres-{label}-backup-bash-command",
                                     bash_command=bash_command,
                                     dag=dag,
                                     append_env=True,
                                     env=env,
                                     owner='abigail',
                                     doc='makes the backup operation and will stored in the directory')

    # backedup_file_sensor = FileSensor(task_id=f'verify-{label}-backedup-file',
    #                                   retries=1,
    #                                   soft_fail=True,
    #                                   max_wait=2,
    #                                   filepath=filename)

    trim_directory_args = {
        'path': backup_directory,
        'filter': '*.sql.gz',
        'preserve_count': preserve_files
    }
    
    trim_directory_op = PythonOperator(task_id=f'trim-{label}-backup-directory',
                                       python_callable=trim_directory,
                                       op_kwargs=trim_directory_args,
                                       dag=dag,
                                       owner='abigail',
                                       retries=1,
                                       doc='preserves only n most recent files, all other files will removed')

    prepare_directory_op >> backup_command_op >> trim_directory_op

    logging.info('daily backup queued up')

if __name__ == "__main__":
    dag.test()
