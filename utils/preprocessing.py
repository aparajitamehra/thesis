"""This module contains functions related to data preprocessing."""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class HighVIFDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops numerical columns with high variance inflation factor."""

    def __init__(self, threshold=10):
        self.threshold = threshold
        self._high_vif_cols = []

    def fit(self, X, y=None):
        """Identifies columns with a VIF greater than `self.threshold`."""
        self._identify_high_vif(X)
        print(f"Dropping columns '{', '.join([str(i) for i in self._high_vif_cols])}'")
        return self

    def transform(self, X, y=None):
        """Drops columns identified in the fit method."""
        return np.delete(X, self._high_vif_cols, axis=1)

    def _identify_high_vif(self, data):
        """Identifies columns with a variance inflation factor (VIF) over `self.threshold`.

        Params:
            data: an np.ndarray of numerical data with shape (num_samples, num_columns)
        """

        original_indices = [i for i in range(data.shape[1])]
        drop = True
        while drop:
            drop = False
            max_vif = -1
            max_col = None
            for i, col in enumerate(data.T):
                vif = variance_inflation_factor(data, i)
                if vif > max_vif:
                    max_vif = vif
                    max_col = i

            if max_vif > self.threshold:
                self._high_vif_cols.append(original_indices.pop(max_col))
                data = np.delete(data, max_col, axis=1)
                drop = True


class EmbeddingExtractor((BaseEstimator, TransformerMixin)):
    def __init__(self, embedding_trainer, weights=None):
        # self.transform_model = None
        # self.weights = weights
        # if self.weights:
        # try:
        # self.transform_model = keras.model.load_from(weights)
        # except:
        # weights file error during load
        # print("WARNING: could not load weights,
        # will learn embeddings on next fit")
        # self.transform_model = None
        return self

    # def save(self, weights):
    #     if self.transform_model:
    #         self.transform_model.save(weights)

    def fit(self, X, y=None):
        # if self.transform_model:
        # return self
        # else:
        # for each categorical var
        # input_models.append(input_model)
        # embeddings.append(embedding)
        # add numerical stuff too

        # prev_layer = embeddings
        # for i, layer in enumerate(self.embedding_trainer):
        #     self.embedding_trainer[i] = layer(prev_layer)
        #     prev_layer = layer
        # model = Model(inputs=input_models, outputs=self.embedding_trainer[-1])
        # model.compile()
        # model.fit(X)
        # self.transform_model = Model(inputs=input_models, outputs=embeddings)

        return self

    def transform(self, X, y=None):
        # tfs = self.transform_model.predict(X)
        # return np.concatenate(tfs, axis=1)
        return self


def preprocessing_pipeline_onehot(data):
    numeric_features = data.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("vif_dropper", HighVIFDropper(threshold=10)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_features = data.select_dtypes("category").columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor_onehot = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor_onehot


def preprocessing_pipeline_embedding(data):
    numeric_features = data.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("vif_dropper", HighVIFDropper(threshold=10)),
            ("scaler", StandardScaler()),
        ]
    )
    num_transformer = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )
    preprocessor_emb = Pipeline(
        steps=[("column", num_transformer), ("embeddings", EmbeddingExtractor())]
    )
    return preprocessor_emb
