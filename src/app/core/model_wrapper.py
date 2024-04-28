
import enum
from dataclasses import dataclass
import os

import numpy as np
import scipy
import sklearn
from pyparsing import Any
from tensorflow import keras


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
        if not os.path.exists(model_url):
            return None
        self.model = keras.saving.load_model(model_url)
