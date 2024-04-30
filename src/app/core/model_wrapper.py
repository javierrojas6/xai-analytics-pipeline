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
    
    def transform(self, x):
        """
        The function `transform` checks if a vectorizer is instantiated and applies different
        transformation techniques based on the vector type.
        
        :param x: It looks like the code you provided is a method named `transform` within a class, and
        it takes a parameter `x`. The method checks if a vectorizer is instantiated for the class
        instance (`self`) and then transforms the input `x` based on the vectorization technique
        specified by `self
        :return: The `transform` method returns the transformed input `x` based on the vectorization
        technique specified in `self.vector_type`. If the vector type is `N_GRAM`, it transforms the
        input using the vectorizer and returns it as a NumPy array. If the vector type is `SEQUENCE`, it
        directly returns the result of the vectorizer applied to the input. If the vector type is
        """
        if self.vectorizer == None:
            raise Exception(f'vectorizer not instantiated for {self.name}')

        if self.vector_type == VectorizeTechnique.N_GRAM:
            transformed = self.vectorizer.transform(x)
            return np.array(scipy.sparse.csr_matrix.toarray(transformed))

        elif self.vector_type == VectorizeTechnique.SEQUENCE:
            return self.vectorizer(x)

        return None

    def predict(self, x):
        """
        This Python function predicts outcomes using a specified vectorization technique and a machine
        learning model.
        
        :param x: It looks like the `predict` method in the code snippet is used to make predictions
        using a machine learning model. The input parameter `x` is the data that you want to make
        predictions on
        :return: The predict method returns the predictions made by the model on the input data x after
        transforming it using the appropriate vectorization technique specified in the code. The
        predictions are flattened before being returned.
        """
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
