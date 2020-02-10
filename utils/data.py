""" This module provides utility functions for reading/writing data to disk."""

import pandas as pd


def load_credit_scoring_data(data_path, descriptor_path):
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
    return pd.read_csv(
        data_path,
        header=None,
        skipinitialspace=True,
        names=descriptor["Name"],
        dtype=dtype_dict,
    )
