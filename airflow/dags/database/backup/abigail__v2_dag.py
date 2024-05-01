import logging
import os
import shutil
import sys

import pendulum
from dotenv import load_dotenv

from airflow import DAG
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook

sys.path.append(os.path.join(os.getcwd(), './src'))
from app.core.file.folder import trim_directory
from app.core.file.folder import prepare_directory
from app.core.database.backup import build_backup_bash_command


with DAG(dag_id="Abigail.backup.02",
         start_date=pendulum.datetime(2024, 1, 1, tz='America/Bogota'),
         schedule="*/10 * * * *",
         #  schedule="@daily",
         tags=['abigail', 'backup'],
         orientation='TB',
         max_active_tasks=1,
         max_active_runs=1) as dag:

    label = 'hourly'
    backup_directory = f'backups/{label}'
    preserve_files = 10

    @task()
    def task_prepare_folder(directory):
        logging.info('preparing directory')
        prepare_directory(directory)

    @task()
    def task_backup():
        logging.info('backing up')
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

        bash_command, filename = build_backup_bash_command(settings=settings, directory=os.path.join(os.getcwd(), backup_directory))

        logging.info(f'backing up into {bash_command}')

        subprocess = SubprocessHook()
        bash_path = shutil.which("bash") or "bash"
        code, output = subprocess.run_command(command=[bash_path, "-c", bash_command], env=env)
        
        if code == 0:
            logging.info('backup succeed')
        else:
            logging.info(f'backup error {code}, {output}')
            
        return code, output
    
    @task()
    def task_trim_directory(backup_directory, preserve_files):
        logging.info(f'trimming directory {backup_directory}')
        
        trim_directory(path=backup_directory,
                       filter='*.sql.gz',
                       preserve_count=preserve_files)


    logging.info(f'starting {label}')

    task_prepare_folder(backup_directory) \
        >> task_backup() \
        >> task_trim_directory(backup_directory, preserve_files)


if __name__ == "__main__":
    dag.test()