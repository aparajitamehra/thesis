from scipy.stats import stats

from utils.data_loading import load_credit_scoring_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main_data_analysis(data_path, descriptor_path, ds_name):
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )
    print(f"{ds_name}")
    print(f"shape: {X.shape}")
    print(f"types: {X.dtypes.value_counts()}")
    print(f"classes\n {y.value_counts()}")

    for col in X.select_dtypes("number").columns:
        col = f"{col}"
        sns.boxplot(x=X[col])


if __name__ == "__main__":
    from pathlib import Path

    # for each dataset:
    for ds_name in ["bene1", "bene2"]:
        print(ds_name)

        main_data_analysis(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )
