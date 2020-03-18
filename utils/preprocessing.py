"""This module contains functions related to data preprocessing."""
# import keras
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler, KBinsDiscretizer


# from keras.layers import Input, Dense, Activation, Reshape
# from keras.models import Model
# from keras.layers import Concatenate, Dropout
# from keras.layers.embeddings import Embedding


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


"""
def preproc(X_train, X_val, X_test, data_df):
    embed_cols = [i for i in X_train.select_dtypes(include=["category", "bool"])]
    num_cols = [i for i in X_train.select_dtypes(include=["number"])]

    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i, cat in enumerate(raw_vals):
            val_map[cat] = i
        input_list_train.append(X_train[c].map(val_map).fillna(0).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in num_cols]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


class EmbeddingExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_trainer, weights=None):
        self.embedding_trainer = embedding_trainer
        self.transform_model = None
        self.weights = weights
        if self.weights:
            try:
                self.transform_model = keras.model.load_model(weights)
            except:
                print("WARNING: could not load weights,
                will learn embeddings on next fit")
                self.transform_model = None

    def save(self, weights):
        if self.transform_model:
            self.transform_model.save(weights)

    def fit(self, X, y=None):
        if self.transform_model:
            return self
        else:
            target = ["censor"]
            features = X.columns.difference(["censor"])

            X_train, y_train = X.iloc[:21000][features], X.iloc[:21000][target]
            X_val, y_val = X.iloc[21000:27000][features], X.iloc[21000:27000][target]
            X_test = X.iloc[27000:][features]

            embed_cols = [i for i in X_train.select_dtypes(
                    include=["category", "bool"])]
            num_cols = [i for i in X_train.select_dtypes(include=["number"])]

            for i in num_cols:
                X_train[i].values.reshape(-1, 1)
                X_train[i] = X_train[i].values.reshape(-1, 1)
                X_val[i] = X_val[i].values.reshape(-1, 1)

            input_models = []
            output_embeddings = []

            for categorical_var in embed_cols:
                cat_emb_name = categorical_var.replace(" ", "") + "_Embedding"

                no_of_unique_cat = X_train[categorical_var].nunique()
                embedding_size = int(min(np.ceil(no_of_unique_cat / 2), 50))

                input_model = Input(shape=(1,))
                output_model = Embedding(no_of_unique_cat,
                    embedding_size,
                    name=cat_emb_name)(
                    input_model
                )
                output_model = Reshape(target_shape=(embedding_size,))(output_model)

                input_models.append(input_model)
                output_embeddings.append(output_model)

            input_numeric = Input(
                shape=(len(X_train.select_dtypes(include=["number"]).columns.tolist()),)
            )
            embedding_numeric = Dense(128)(input_numeric)
            input_models.append(input_numeric)
            output_embeddings.append(embedding_numeric)

        prev_layer = output_embeddings
        for i, layer in enumerate(self.embedding_trainer):
            self.embedding_trainer[i] = layer(prev_layer)
            prev_layer = layer
        model = Model(inputs=input_models, outputs=self.embedding_trainer[-1])
        model.compile()

        X_train_list, X_val_list, X_test_list = preproc(
            X_train, X_val, X_test)

        model.fit(
            X_train_list,
            y_train,
            validation_data=(X_val_list, y_val),
            epochs=2,
            batch_size=512,
            verbose=2,
        )

        self.transform_model = Model(inputs=input_models, outputs=output_embeddings)
        return self

    def transform(self, X, y=None):
        tfs = self.transform_model.predict(X)
        np.concatenate(tfs, axis=1)
        return self
"""


def preprocessing_pipeline_onehot(data):
    numeric_features = data.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("highVifDropper", HighVIFDropper()),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_features = data.select_dtypes(include=("bool", "category")).columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    onehot_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return onehot_preprocessor


def preprocessing_pipeline_dummy(data):

    categorical_features = data.select_dtypes(include="category").columns
    cat_list = []
    for var in categorical_features:
        cat = "var" + "_" + var
        cat = pd.get_dummies(data[var], prefix=var)
        data1 = data.join(cat)
        data = data1
        cat_list = cat_list + (cat.columns.values.tolist())
    data_vars = data.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in categorical_features]

    data_final = data[to_keep]
    data_final.columns.values

    numeric_features = data_final.columns.difference(cat_list)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("highVifDropper", HighVIFDropper()),
            ("scaler", StandardScaler()),
        ]
    )

    dummy_preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )

    return dummy_preprocessor


def MLPpreprocessing_pipeline_onehot(data):
    numeric_features = data.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("highVifDropper", HighVIFDropper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_features = data.select_dtypes(include=("bool", "category")).columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    onehot_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return onehot_preprocessor


def cnn2dprep_num(data,binsize):
    numeric_features = data.select_dtypes("number").columns
    print("n_numeric: ", numeric_features.size)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("highVifDropper", HighVIFDropper()),
            ("Standardizer", StandardScaler()),
            ("discretizer", KBinsDiscretizer(n_bins=binsize, strategy='uniform',encode='onehot-dense'))
        ]
    )
    onehot_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
        ]
    )

    return onehot_preprocessor


"""
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
    embedding_trainer = [Concatenate,
                         Dense(1000, kernel_initializer="uniform"),
                         Activation("relu"),
                         Dropout(0.4),
                         Dense(512, kernel_initializer="uniform"),
                         Activation("relu"),
                         Dropout(0.3),
                         Dense(1, activation="sigmoid")]
    emb_preprocessor = Pipeline(
        steps=[("column", num_transformer),
        ("embeddings", EmbeddingExtractor(embedding_trainer=embedding_trainer))]
    )
    return emb_preprocessor
"""
