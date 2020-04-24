import numpy as np
from sklearn.externals import joblib
from keras.models import load_model
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.data_loading import load_credit_scoring_data
from sklearn.metrics import (
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    classification_report,
)


def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)


def test_scores_sklearn(data_path, descriptor_path, model, classifier, ds_name):
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    model = joblib.load(model)
    model.fit(X_train, y_train)

    with open(f"{classifier}_SCORES_{ds_name}.txt", "w") as f:
        f.write(f"Best auc_score on train set: {model.score(X_test, y_test):.3f}\n")
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, model.predict(X_train))}\n"
        )

        # get score and classification report for test data
        f.write(
            f"Scores for test set: {classification_report(y_test, model.predict(X_test))}\n"
        )
        f.write(
            f"f1 score on test set: {f1_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(f"fbeta_score on test set: {f2(y_test, model.predict(X_test)):.3f}\n")
        f.write(
            f"AUC of test set: {roc_auc_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(
            f"Accuracy of test set: {accuracy_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(
            f"Precision of test set: {precision_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(
            f"Recall of test set: {recall_score(y_test, model.predict(X_test)):.3f}\n"
        )


def test_scores_keras(data_path, descriptor_path, model, classifier, ds_name):
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    model = load_model(f"{model}_{ds_name}.h5")
    model.fit(X_train, y_train)

    with open(f"{classifier}_SCORES_{ds_name}.txt", "w") as f:
        f.write(f"Best auc_score on train set: {model.score(X_test, y_test):.3f}\n")
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, model.predict(X_train))}\n"
        )

        # get score and classification report for test data
        f.write(
            f"Scores for test set: {classification_report(y_test, model.predict(X_test))}\n"
        )
        f.write(
            f"f1 score on test set: {f1_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(f"fbeta_score on test set: {f2(y_test, model.predict(X_test)):.3f}\n")
        f.write(
            f"AUC of test set: {roc_auc_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(
            f"Accuracy of test set: {accuracy_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(
            f"Precision of test set: {precision_score(y_test, model.predict(X_test)):.3f}\n"
        )
        f.write(
            f"Recall of test set: {recall_score(y_test, model.predict(X_test)):.3f}\n"
        )


# if __name__ == "__main__":
#     from pathlib import Path
#
#     # for each dataset:
#     for ds_name in ["UK"]:
#         print(ds_name)
#         # define embedding model saved model file
#
#         for classifier in ["logreg", "mlp_sklearn", "ranfor", "xgboost"]:
#             test_scores(
#                 f"datasets/{ds_name}/input_{ds_name}.csv",
#                 f"datasets/{ds_name}/descriptor_{ds_name}.csv",
#                 f"models/{classifier}_{ds_name}.pkl",
#                 classifier,
#                 ds_name,
#             )


def HighVif(data_path, desc_path, ds_name):

    data, _, _, _, _, _ = load_credit_scoring_data(data_path, desc_path)

    data = data.select_dtypes("number").values
    threshold = 10

    high_vif_cols = []
    original_indices = [i for i in range(data.shape[1])]
    x = 0
    while x < 2:
        max_vif = -1
        max_col = None
        vifs = []
        for i, col in enumerate(data.T):
            vif = variance_inflation_factor(data, i)
            vifs.append(vif)
            if vif > max_vif:
                max_vif = vif
                max_col = i
        print(f"{ds_name}: {vifs}")

        if max_vif > threshold:
            high_vif_cols.append(original_indices.pop(max_col))
            data = np.delete(data, max_col, axis=1)
        x += 1


if __name__ == "__main__":
    # for each dataset:
    for ds_name in ["UK", "bene1", "bene2", "german"]:

        HighVif(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )
