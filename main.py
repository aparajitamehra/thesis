import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    make_scorer,
    recall_score,
    roc_curve,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    auc,
)

from utils.data import load_credit_scoring_data
from utils.preprocessing import (
    preprocessing_pipeline_onehot,
    preprocessing_pipeline_dummy,
)
import numpy as np
from kerasformain import makeweightedMLP,plot_metrics, make_weightedCNN, make_weighted2dCNN,make_weighted_hybrid_CNN


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
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}

key_results = [
    "mean_test_recall_score",
    "mean_test_f1_score",
    "mean_test_accuracy_score",
    "mean_test_my_auc",
    "mean_test_balanced_accuracy",
]


def xgbtuning(classifier):
    od = {
        'max_depth': list(range(3, 10, 2)),

        'learning_rate': list(np.arange(0.01, 0.1, 0.01)),

        'gamma': list([i/10.0 for i in range(0,5)]),
    }

    gridxgb = GridSearchCV(
        estimator=classifier,
        param_grid=od,
        n_jobs=-1,
        cv=5,
        scoring = scorers,
        refit = 'my_auc',
        verbose=1
    )


    return gridxgb


def lggridsearch(classifier):
    grid_lg = {
        "penalty": ["l1", "l2", "elasticnet"],
        "dual": [True, False],
        "C": np.logspace(-1, 1, 7),
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    }

    gridlg = GridSearchCV(
        estimator=classifier,
        param_grid=grid_lg,
        cv=5,
        n_jobs=-1,
        scoring=scorers,
        refit="balanced_accuracy",
        verbose=1,
        return_train_score=True,
    )
    return gridlg


def rfgridsearch(classifier):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=600, stop=1200, num=5)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [2, 5, 8]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 5, 8]
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
        cv=2,
        n_jobs=-1,
        scoring=scorers,
        refit="balanced_accuracy",
        verbose=1,
    )

    return gridrf


def mlpgridsearch(classifier):
    grid_mlp = {
        "batch_size": [100, 700, "auto"],
        "hidden_layer_sizes": [(5, 2), (20, 2), (75, 2)],
        "activation": ["tanh"],
        "solver": ["lbfgs"],
        "alpha": [0.001],
        "momentum": [0.8, 1.0],
        "learning_rate": ["constant", "adaptive"],
    }

    gridmlp = GridSearchCV(
        estimator=classifier,
        param_grid=grid_mlp,
        n_jobs=-1,
        cv=5,
        scoring=scorers,
        refit="balanced_accuracy",
        verbose=1,
    )

    return gridmlp


def main(data_path, descriptor_path, ds_name):
    data = load_credit_scoring_data(data_path, descriptor_path)
    y = data.pop("censor")
    X = data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )



    onehot_preprocessor = preprocessing_pipeline_onehot(X_train)
    dummy_preprocessor = preprocessing_pipeline_dummy(X_train)

    log_reg = create_classifier(
        dummy_preprocessor,
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

    XGboost = create_classifier(
        onehot_preprocessor,
        xgbtuning(XGBClassifier(
            silent=False,
            scale_pos_weight=1,
            learning_rate=0.01,
            colsample_bytree = 0.4,
            subsample = 0.8,
            objective='binary:logistic',
            reg_alpha = 0.3,
            gamma=10
        ))


    )

    random_forest = create_classifier(
        onehot_preprocessor,
        rfgridsearch(RandomForestClassifier(random_state=42, class_weight="balanced")),
    )

    mlp = create_classifier(
        onehot_preprocessor,
        mlpgridsearch(MLPClassifier(max_iter=5000, early_stopping=True)),
    )
    '''
    sklclassifiers = {
        "LogisticRegressionCV": log_reg,
        "RandomForest": random_forest,
        "AdaBoostClassifier": adaboost,
        "MLP": mlp,
    }
    '''
    sklclassifiers = {
        "XGboost": XGboost
    }
    '''
    for name, clf in sklclassifiers.items():
        clf.fit(X_train, y_train)
        print(f'{name} Best parameters found: {clf["classifier"].best_params_}')
        print(f'{name} score: {clf["classifier"].best_score_}')
        pd.DataFrame(clf["classifier"].cv_results_)[key_results].to_csv(
            f"{name}_results.csv"
        )

        X_test_tr = clf["preprocessor"].transform(X_test)
        y_pred = clf["classifier"].predict(X_test_tr)

        labels = [0, 1]
        print(confusion_matrix(y_test, y_pred, labels=labels))
        print(classification_report(y_test, y_pred))

    '''

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    weightedMLP =makeweightedMLP(X_train, X_test, y_train, y_test)
    plt.figure(ds_name)
    plot_metrics(weightedMLP,'MLP',colors[0])
    cnnhist= make_weightedCNN(X_train, X_test, y_train, y_test)
    plot_metrics(cnnhist,'CNN', colors[1])

    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hybrid=make_weighted_hybrid_CNN(X_train, X_test, y_train, y_test)
    plt.figure(ds_name)
    plot_metrics(hybrid, 'hybrid', colors[2])

if __name__ == "__main__":

    #for ds_name in ["UK", "bene1", "bene2", "german"]:
    for ds_name in ["UK"]:
        print(ds_name)
        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )

    plt.show()
