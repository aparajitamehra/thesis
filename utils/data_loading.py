""" This module provides utility functions for reading/writing data to disk."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_credit_scoring_data(data_path, descriptor_path, rearrange=False):
    """Loads credit scoring data from a CSV file"""

    # define column names and types
    descriptor = pd.read_csv(descriptor_path)
    dtype_dict = dict(zip(descriptor["Name"], descriptor["Type"]))

    # read data
    data = pd.read_csv(
        data_path,
        header=None,
        skipinitialspace=True,
        names=descriptor["Name"],
        dtype=dtype_dict,
    )

    # output rearranged columns for CNNs
    if rearrange:

        target = data.pop("censor")
        num_var = data[data.select_dtypes("number").columns]
        cat_var = data[data.select_dtypes(include=["category", "bool"]).columns]
        num_data = pd.concat([num_var, target], axis=1)
        ix = (
            num_data.corr()
            .sort_values("censor", ascending=False, na_position="last")
            .index
        )
        num_data = num_data.loc[:, ix]

        data = pd.concat([num_data, cat_var], axis=1)

        y = data.pop("censor")
        X = data
        print("rearrangetrue")

    # output original, non rearranged data for all others
    else:
        y = data.pop("censor")
        X = data
        print("rearearrangefalse")

    # split train, test data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.1, random_state=5
    )

    return X, y, X_train, X_test, y_train, y_test
