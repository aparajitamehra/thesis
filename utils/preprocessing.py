"""This module contains functions related to data preprocessing."""

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor


class HighVIFDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops numerical columns with high variance inflation factor."""

    def __init__(self, threshold=10):
        self._threshold = threshold

    def fit(self, X, y=None):
        """Does nothing for the HighVIFDropper."""
        return self

    def transform(self, X, y=None):
        """Drops columns in X with a VIF greater than `self._threshold`."""
        return self._drop_high_vif(X, self._threshold)

    @staticmethod
    def _drop_high_vif(data, threshold=10):
        """Drops numerical columns with a variance inflation factor (VIF) over `threshold`.

        Params:
            data: a pandas DataFrame
            threshold: columns with a VIF higher than this value will be dropped
        """

        drop = True
        while drop:
            drop = False
            numeric_cols = data.select_dtypes("number").columns
            max_vif = -1
            max_col = None
            for i, col in enumerate(numeric_cols):
                vif = variance_inflation_factor(data[numeric_cols].values, i)
                if vif > max_vif:
                    max_vif = vif
                    max_col = col

            if max_vif > threshold:
                print(f"Dropping column '{max_col}' with VIF: {max_vif}")
                data = data.drop([max_col], axis=1)
                drop = True

        return data
