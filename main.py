from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score, roc_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    auc,
)

from utils.data import load_credit_scoring_data
from utils.preprocessing import preprocessing_pipeline_onehot
import numpy as np


def create_classifier(preprocessor, classifier):
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return clf


def custom_auc(x, y):
    fpr, tpr, _ = roc_curve(x, y, pos_label=1)
    return auc(fpr, tpr)


# define my_auc using custom function
my_auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)

scorers = {
    "recall_score": make_scorer(recall_score),
    "f1_score": make_scorer(f1_score),
    "accuracy_score": make_scorer(accuracy_score),
    "my_auc": my_auc,
}


def lggridsearch(classifier):
    grid_lg = {
        "penalty": ["l1", "l2", "elasticnet"],
        "dual": [True, False],
        "C": np.logspace(0, 5, 7),
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    }

    gridlg = GridSearchCV(
        estimator=classifier,
        param_grid=grid_lg,
        cv=2,
        n_jobs=-1,
        scoring=scorers,
        refit="f1_score",
    )
    return gridlg


def rfgridsearch(classifier):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=11)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [5, 8, 15, 25, 30]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    grid_rf = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    gridrf = GridSearchCV(
        estimator=classifier,
        param_grid=grid_rf,
        cv=5,
        n_jobs=-1,
        scoring=scorers,
        refit="f1_score",
    )

    # bestrf = gridrf.fit(X_train, y_train)
    return gridrf


def mlpgridsearch(classifier):
    grid_mlp = {
        "batch_size": [10, 100, 1000, "auto"],
        "hidden_layer_sizes": [
            (5, 2),
            (20, 2),
            (50, 2),
            (75, 2),
            (100, 2),
            (50, 20, 2),
            (20, 20, 2),
            (10, 10, 2),
        ],
        "activation": ["relu", "tanh"],
        "solver": ["sgd", "lbfgs", "adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "momentum": [0.6, 0.8, 1.0],
        "learning_rate": ["constant", "adaptive"],
    }

    gridmlp = GridSearchCV(
        estimator=classifier,
        param_grid=grid_mlp,
        n_jobs=-1,
        cv=2,
        scoring=scorers,
        refit="f1_score",
    )

    return gridmlp


def main(data_path, descriptor_path):
    data = load_credit_scoring_data(data_path, descriptor_path)

    y = data.pop("censor")
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    onehot_preprocessor = preprocessing_pipeline_onehot(X_train)

    log_reg = create_classifier(
        onehot_preprocessor,
        lggridsearch(LogisticRegression(max_iter=5000, class_weight="balanced",),),
    )

    adaboost = create_classifier(
        onehot_preprocessor,
        AdaBoostClassifier(
            RandomForestClassifier(
                random_state=42,
                n_estimators=500,
                n_jobs=-1,
                oob_score=True,
                class_weight="balanced",
                max_depth=5,
            ),
            n_estimators=10,
        ),
    )

    random_forest = create_classifier(
        onehot_preprocessor,
        rfgridsearch(RandomForestClassifier(random_state=42, class_weight="balanced")),
    )

    mlp = create_classifier(
        onehot_preprocessor,
        mlpgridsearch(MLPClassifier(max_iter=5000, early_stopping=True)),
    )

    classifiers = {
        "LogisticRegressionCV": log_reg,
        "RandomForest": random_forest,
        "AdaBoostClassifier": adaboost,
        "MLP": mlp,
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        print(f'{name} Best parameters found: {clf["classifier"].best_params_}')
        print(f'{name} f1 score: {clf["classifier"].best_score_}')

        X_test = onehot_preprocessor.fit_transform(X_test)
        y_pred = clf["classifier"].predict(X_test)
        labels = [0, 1]
        print(confusion_matrix(y_test, y_pred, labels=labels))
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    for ds_name in ["german"]:
        print(ds_name)
        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
        )
