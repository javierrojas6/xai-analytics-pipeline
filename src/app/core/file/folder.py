import os
import numpy as np
import glob


def prepare_directory(path, mode: int = 0o777):
    """
    The function `prepare_directory` creates a directory at the specified path if it does not already
    exist.

    :param path: The `path` parameter is a string that represents the directory path that needs to be
    prepared. It can be an absolute or relative path
    :param mode: The `mode` parameter is an optional argument that specifies the permissions for the
    newly created directories. It is an integer value that represents the octal value of the
    permissions. By default, it is set to `0o777`, which means the directories will have read, write,
    and execute permissions for, defaults to 0o777
    :type mode: int (optional)
    :return: nothing (None).
    """
    if os.path.exists(path):
        return

    segments = list(filter(lambda x: len(x) > 0, path.split('/')))

    partial_url_segments = []
    partial_path = ''
    for segment in segments:
        partial_url_segments += [segment]

        partial_path = '/'.join(partial_url_segments)

        if not os.path.exists(partial_path):
            os.mkdir(partial_path, mode)


def trim_directory(path: str, filter: str = '*', preserve_count: int = 30):
    """
    The `trim_directory` function removes files from a specified directory based on a filter and
    preserves a specified number of the most recent files.
    
    :param path: The `path` parameter is a string that represents the directory path where the files are
    located
    :type path: str
    :param filter: The "filter" parameter is used to specify a pattern for filtering the files in the
    directory. It uses the same pattern matching syntax as the Unix shell. For example, if you want to
    filter only the text files, you can set the filter parameter to "*.txt", defaults to *
    :type filter: str (optional)
    :param preserve_count: The `preserve_count` parameter determines the number of files that should be
    preserved in the directory. If the number of files in the directory is greater than
    `preserve_count`, the function will remove the excess files, defaults to 30
    :type preserve_count: int (optional)
    :return: If the number of files in the directory is less than or equal to the `preserve_count`,
    nothing is returned. Otherwise, the function removes the excess files and does not return anything.
    """
    files = glob.glob(os.path.join(path, filter))

    if len(files) <= preserve_count:
        return

    files = list(zip(files, map(lambda x: os.path.getctime(x), files)))
    files.sort(key=lambda x: x[1], reverse=True)

    for f in np.array(files)[preserve_count:, 0]:
        os.remove(f)
