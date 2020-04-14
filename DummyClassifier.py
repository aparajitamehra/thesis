from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from utils.data import load_credit_scoring_data
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
    average_precision_score,
)


# define fbeta metric with beta = 3
def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)


scorers = {
    "recall_score": make_scorer(recall_score),
    "f1_score": make_scorer(f1_score),
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score),
    "fbeta_score": make_scorer(f2),
    "auc": make_scorer(roc_auc_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "average_precision": make_scorer(average_precision_score),
}


def main_dummy(data_path, descriptor_path, ds_name):

    # load and split data
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    def dummy_tuning(classifier):
        params = {"strategy": ["most_frequent"]}

        griddummy = GridSearchCV(
            estimator=classifier,
            param_grid=params,
            cv=5,
            scoring=scorers,
            refit="average_precision",
        )

        return griddummy

    dummy_clf = dummy_tuning(DummyClassifier())
    dummy_clf.fit(X_train, y_train)

    preds = dummy_clf.predict(X_test)

    with open(f"dummy_results_{ds_name}.txt", "w") as f:
        f.write(f"Best auc on train set: {dummy_clf.best_score_:.3f}\n")
        f.write(f"Best parameter set: {dummy_clf.best_params_}\n")
        f.write(f"Best scores index: {dummy_clf.best_index_}\n")
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, dummy_clf.predict(X_train))}\n"
        )

        # get score and classification report for test data
        f.write(f"Scores for test set: {classification_report(y_test, preds)}\n")
        f.write(f"f1 score on test set: {f1_score(y_test, preds):.3f}\n")
        f.write(f"fbeta_score on test set: {f2(y_test, preds):.3f}\n")
        f.write(f"AUC of test set: {roc_auc_score(y_test, preds):.3f}\n")
        f.write(f"Accuracy of test set: {accuracy_score(y_test, preds):.3f}\n")
        f.write(f"Precision of test set: {precision_score(y_test, preds):.3f}\n")
        f.write(f"Recall of test set: {recall_score(y_test, preds):.3f}\n")
        f.write(
            f"Average Precision of test set: {average_precision_score(y_test, preds)}\n"
        )


if __name__ == "__main__":
    for ds_name in ["UK", "bene1", "bene2", "german"]:
        print(ds_name)
        main_dummy(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )
