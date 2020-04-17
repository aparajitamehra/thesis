""" This module provides utility functions for reading/writing data to disk."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_credit_scoring_data(data_path, descriptor_path, rearrange=False):
    """Loads credit scoring data from a CSV file.

    Params:
        data_path: filename of main data
        descriptor_path: filename of field descriptor file. Must contain at a minimum a
            column "Name" with the names of the fields and a column "Type" with the
            pandas dtype of that field

    Returns:
        A pandas DataFrame of the data in `data_path` with descriptors set per the
        contents of `descriptor_path`.
    """

    descriptor = pd.read_csv(descriptor_path)
    dtype_dict = dict(zip(descriptor["Name"], descriptor["Type"]))

    data = pd.read_csv(
        data_path,
        header=None,
        skipinitialspace=True,
        names=descriptor["Name"],
        dtype=dtype_dict,
    )



    if rearrange == True:
        #print(data.corr().sort_values('censor', ascending=False, na_position='last'))

        print(data.dtypes)
        print(data.head())
        initial_ix= data.columns
        print("init",initial_ix)

        new_ix = data.corr().sort_values('censor', ascending=False, na_position='last').index
        diff = initial_ix.difference(new_ix)

        print("diff: ",diff)
        new_ix=new_ix.append(diff)

        print("new",new_ix)

        X = data.loc[:, new_ix]
        y = X.pop("censor")
        print("rearrangetrue")
        #print(X.head())
        #print(data.head())

    else:
        y = data.pop("censor")
        X = data
        print("rearearrangefalse")


    # split train, test data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=10
    )

    return X, y, X_train, X_test, y_train, y_test
