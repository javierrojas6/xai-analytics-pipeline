from datetime import datetime
import logging


def build_backup_bash_command(settings: dict, directory: str) -> str:
    """
    The function `build_backup_bash_command` builds a bash command for backing up a PostgreSQL database.
    
    :param settings: The `settings` parameter is a dictionary that contains various settings for the
    backup command. It is assumed to have a key-value pair for the `database` setting, which specifies
    the name of the database to be backed up. Other settings can be included as additional key-value
    pairs
    :type settings: dict
    :param directory: The `directory` parameter is a string that represents the directory where the
    backup file will be saved
    :type directory: str
    :return: a bash command string that can be used to create a backup of a PostgreSQL database. The
    command includes the specified settings and database name, and the backup file will be saved in the
    specified directory with a filename that includes the database name and the current date and time.
    """
    logging.info('build_backup_bash_command starting')
    database = settings['database']
    del settings['database']

    command_segments = [f"{str(key)} {str(settings[key])}" for key in settings]
    bash_command_settings = " ".join(command_segments).replace('  ', ' ')
    date_string = datetime.now().strftime('%Y%m%d%H%M')
    filename = f'{directory}/{database}_{date_string}.sql.gz'

    return f'pg_dump {bash_command_settings} {database} | gzip -9 -c > {filename}', filename
