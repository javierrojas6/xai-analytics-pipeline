# %%
import os
import sys
import pandas as pd
import os
from pyparsing import Any
import numpy as np

import tensorflow as tf
from tensorflow import keras


# %%
print(tf.version.VERSION)
sys.path.append(os.path.join(os.getcwd(), '../src'))
from app.benchmark_load_data import load_imdb_sentiment_analysis_dataset
# %%
models_folder = "../models"
performance_report_url = "../performance_report.csv"

# %%
def load_model(model_folder: str, model_name: str) -> Any:
    model_url = os.path.join(model_folder, model_name)
    if not os.path.exists(model_url):
        return None
    
    return keras.saving.load_model(model_url)
# %%
loaded_model = load_model(models_folder, 'mlp_model_v32.h5')

# %%
loaded_model.summary()
# %%
df_performance_report = pd.read_csv(performance_report_url)
# %%
df_performance_report.info()
# %%
df_performance_report.head()

# %%
dataset = load_imdb_sentiment_analysis_dataset('../dataset')
# %%
dataset = np.array(dataset)
# %%
dataset.shape
# %%
train, test = dataset
# %%
x_train, y_train = train
x_test, y_test = test
# %%
print('train len:', len(y_train))
print('test len:', len(y_test))
# %%
print('labels', (np.unique(y_test)))

# %%
models = []
for i, row in df_performance_report[df_performance_report['dataset_name'] == 'IMDB'].iterrows():
    models += [load_model(models_folder, row['model_name'])]
models = np.array(models)    
# %%
models

# %%