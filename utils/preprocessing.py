"""This module contains functions related to data preprocessing."""

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
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


def create_preprocessing_pipeline(data):
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor
