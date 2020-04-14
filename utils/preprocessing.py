"""This module contains functions related to data preprocessing."""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    OneHotEncoder,
    StandardScaler,
    KBinsDiscretizer,
    OrdinalEncoder,
)


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
        x = 0
        while x < 2:
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
            x += 1


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


def EntityPrep(data):

    numeric_features = data.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("highVifDropper", HighVIFDropper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_features = data.select_dtypes(include=("bool", "category")).columns

    # new
    encoding_cats = [sorted(data[i].unique().tolist()) for i in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoder", "passthrough"),
        ]
    )

    entity_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return entity_preprocessor


def cnn2dprep_num(data, binsize):
    numeric_features = data.select_dtypes("number").columns
    print("n_numeric: ", numeric_features.size)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("highVifDropper", HighVIFDropper()),
            ("Standardizer", StandardScaler()),
            (
                "discretizer",
                KBinsDiscretizer(
                    n_bins=binsize, strategy="uniform", encode="onehot-dense"
                ),
            ),
        ]
    )
    onehot_preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )

    return onehot_preprocessor
