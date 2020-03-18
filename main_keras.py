import matplotlib.pyplot as plt
from utils.data import load_credit_scoring_data
from kerasformain import (
    makeweightedMLP,
    plot_metrics,
    make_weightedCNN,
    # make_weighted2dCNN,
    make_weighted_hybrid_CNN,
)


def main(data_path, descriptor_path, ds_name):

    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    weightedMLP = makeweightedMLP(X_train, X_test, y_train, y_test)
    plt.figure(ds_name)
    plot_metrics(weightedMLP, "MLP", colors[0])
    cnnhist = make_weightedCNN(X_train, X_test, y_train, y_test)
    plot_metrics(cnnhist, "CNN", colors[1])

    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hybrid = make_weighted_hybrid_CNN(X_train, X_test, y_train, y_test)
    plt.figure(ds_name)
    plot_metrics(hybrid, "hybrid", colors[2])


if __name__ == "__main__":
    for ds_name in ["UK"]:
        print(ds_name)
        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )

    plt.show()
