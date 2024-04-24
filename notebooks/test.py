#%%
import os
import glob
import numpy as np

# path = 'mnt/d/Users/Javier/Documents/Proyectos/2023/analytic-pipeline/backups/daily'
path = '../backupsx/daily'
max = 10
#%%
def trim_directory(path: str, filter: str = '*', preserve_count: int = 30):
    files = glob.glob(os.path.join(path, filter))
    files = list(zip(files, map(lambda x: os.path.getctime(x), files)))
    files.sort(key=lambda x: x[1], reverse=True)

    for f in np.array(files)[preserve_count:, 0]:
        os.remove(f)


#%%
trim_directory(path, max)