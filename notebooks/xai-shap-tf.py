# %%
import os
import sys
import pandas as pd
import os
import numpy as np
from ast import literal_eval

from sklearn.metrics import classification_report
import tensorflow as tf
import shap

from tqdm import tqdm
# %%
print(tf.version.VERSION)
# %%
sys.path.append(os.path.join(os.getcwd(), '..', 'src'))
from app.benchmark_load_data import load_imdb_sentiment_analysis_dataset
from app.core.model_wrapper import ModelWrapper
# %%
models_folder = "../models"
performance_report_url = "../performance_report.csv"
# %%
df_performance_report = pd.read_csv(performance_report_url)
# %%
df_performance_report.info()
# %%
df_performance_report.head()
# %%
models = []
with tqdm(total=df_performance_report.shape[0], ncols=100, desc='instancing models...') as pbar:
    for i, row in df_performance_report.iterrows():
        pbar.set_description('instantiating: "%s"' % row['model_name'])

        model = ModelWrapper(
            name=row['model_name'],
            model_folder=models_folder,
            vectorization_hyperparameters=literal_eval(row['vectorization_hyperparameters']),
            model_hyperparameters=literal_eval(row['model_hyperparameters']),
        )

        model.build_model()
        models += [model]

        pbar.update(1)

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
y_train = np.array(list(map(int, y_train))) == 1
y_test = np.array(list(map(int, y_test))) == 1
# %%
x_train
# %%
y_train
# %%
print('train len:', len(y_train))
print('test len:', len(y_test))
# %%
print('labels', (np.unique(y_test)))
# %%
with tqdm(total=df_performance_report.shape[0], ncols=100, desc='building vectorizers...') as pbar:
    for i, model in enumerate(models):
        pbar.set_description('vectorizer: "%s"' % model.name)

        model.build_vectorizer(x_train)

        if model.model is not None:
            model.model.summary()

        pbar.update(1)
# %%
model_index = 1
max_display = 50
prediction_index = 2
# %%
predictions1 = models[model_index].predict(x_test)
# %%
print('AVG:', np.average(predictions1))
print('STD:', np.std(predictions1))
print('VAR:', np.var(predictions1))
print('MEDIAN:', np.median(predictions1))
print('COV:', np.cov(predictions1))
# %%
y_predicted = predictions1 > 0.5
# %%
print(classification_report(y_test, y_predicted, target_names=['negative', 'positive']))
# %%
x_train[prediction_index], y_train[prediction_index], y_predicted[prediction_index], predictions1[prediction_index]
# %%
explainer = shap.Explainer(models[model_index].predict, shap.maskers.Text(r"\W"))
# %%
shap_values = explainer(x_train[prediction_index - 1 : prediction_index])
# %%
shap_values.base_values
# %%
print('score: ', predictions1[prediction_index])
# %%
shap.initjs()
# %%
shap.plots.text(shap_values=shap_values, separator=' ')
# %%
shap.plots.waterfall(shap_values[0], max_display=max_display)
# %%
shap.plots.bar(shap_values[0], max_display=max_display, clustering_cutoff=2)
# %%
shap.plots.force(shap_values[0])
