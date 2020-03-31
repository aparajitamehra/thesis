import matplotlib.pyplot as plt
import seaborn as sns
from utils.data import load_credit_scoring_data
from imblearn.over_sampling import RandomOverSampler
from kerasformain import (
    makeweightedMLP,
    plot_metrics,
    make_weightedCNN,
    # make_weighted2dCNN,
    make_weighted_hybrid_CNN,
    make_weighted2dCNN,
    make_tuned_MLP,
    plot_cm,
    plot_model
)
import csv
from keras.models import load_model

from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    classification_report,
    balanced_accuracy_score,
)
from utils.entity_embedding import EntityEmbedder

def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)



def main(data_path, descriptor_path,  embedding_model, ds_name,):

    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    # oversampling
    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    weightedMLP = makeweightedMLP(X_train, X_test, y_train, y_test, ds_name)
    plt.figure(ds_name)
    plot_metrics(weightedMLP, "MLP", colors[0])

    weightedCNN = make_weightedCNN(X_train, X_test, y_train, y_test, ds_name)
    plot_metrics(weightedCNN, "CNN", colors[1])


    hybrid = make_weighted_hybrid_CNN(X_train, X_test, y_train, y_test, ds_name)
    plt.figure(ds_name)
    plot_metrics(hybrid, "hybrid", colors[2])


    twodcnn = make_weighted2dCNN(X_train, X_test, y_train, y_test, ds_name)
    fig=plt.figure(ds_name)
    plot_metrics(twodcnn, "2DCNN", colors[3])
    plt.savefig('results_plots/keras_plots/{}_history_plot.png'.format(ds_name))



    #tuned = make_tuned_MLP(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    from pathlib import Path
    for ds_name in ["UK", "bene1", "bene2", "german"]:
        print(ds_name)

        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )

    plt.show()
