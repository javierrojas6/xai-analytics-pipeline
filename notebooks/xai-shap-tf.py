# %%
import enum
import os
import sys
import pandas as pd
import os
from pyparsing import Any
import numpy as np
from ast import literal_eval

import scipy
import sklearn
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import shap  # package used to calculate Shap values
from dataclasses import dataclass

from tqdm import tqdm

# %%
print(tf.version.VERSION)
# %%
sys.path.append(os.path.join(os.getcwd(), '../src'))
from app.benchmark_load_data import load_imdb_sentiment_analysis_dataset
# %%
models_folder = "../models"
performance_report_url = "../performance_report.csv"


# %%
class ModelType (enum.Enum):
    KERNEL = 'kernel'
    MPL = 'mpl'
    TRANSFORMER = 'transformer'


class VectorizeTechnique (enum.Enum):
    N_GRAM = 'n-gram'
    SEQUENCE = 'sequence'


@dataclass
class ModelWrapper():
    name: str = None
    model: Any = None
    model_folder: str = None
    vectorization_hyperparameters: dict = None
    model_hyperparameters: dict = None
    vectorizer: Any = None
    vector_type: VectorizeTechnique = None

    def predict(self, x):
        if self.model == None:
            return None

        if self.vector_type == VectorizeTechnique.N_GRAM:
            transformed = self.vectorizer.transform(x)
            transformed = np.array(scipy.sparse.csr_matrix.toarray(transformed))

        elif self.vector_type == VectorizeTechnique.SEQUENCE:
            transformed = self.vectorizer(x)
            
        return self.model.predict(transformed).flatten()

    def build_vectorizer(self, x):
        if self.vectorizer is not None:
            return

        if self.vectorization_hyperparameters['vectorize_technique'] == VectorizeTechnique.N_GRAM.value:
            self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                analyzer=self.vectorization_hyperparameters['TOKEN_MODE'],
                min_df=self.vectorization_hyperparameters['MIN_DOCUMENT_FREQUENCY'],
                ngram_range=self.vectorization_hyperparameters['NGRAM_RANGE'],
                max_features=self.vectorization_hyperparameters['TOP_K'])

            self.vector_type = VectorizeTechnique.N_GRAM
            self.vectorizer.fit_transform(x)

        elif self.vectorization_hyperparameters['vectorize_technique'] == VectorizeTechnique.SEQUENCE.value:
            self.vectorizer = keras.layers.TextVectorization(
                standardize='lower_and_strip_punctuation',
                pad_to_max_tokens=True,
                max_tokens=self.vectorization_hyperparameters['TOP_K'],
                output_sequence_length=self.vectorization_hyperparameters['MAX_SEQUENCE_LENGTH'])

            self.vector_type = VectorizeTechnique.SEQUENCE
            self.vectorizer.adapt(x)

    def build_model(self):
        model_url = os.path.join(self.model_folder, self.name)
        if not os.path.exists(model_url): return None
        self.model = keras.saving.load_model(model_url)

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
predictions1 = models[0].predict(x_test)
# %%
y_predicted = predictions1 > 0.5
# %%
print(classification_report(y_test, y_predicted, target_names=['negative', 'positive']))
# %%
x_train[:2], y_train[:2]
# %%
explainer = shap.Explainer(models[0].predict, shap.maskers.Text(r"\W"))

# %%
shap_values = explainer(x_train[:1])

# %%
shap_values
# %%
shap.initjs()
# %%
shap.plots.text(shap_values=shap_values[0], separator=' ')
# %%
shap.plots.waterfall(shap_values[0])
# %%
shap.plots.bar(shap_values[0], max_display=50, clustering_cutoff=2)