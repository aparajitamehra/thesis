import pandas as pd
from sklearn.linear_model import LogisticRegression

from keras.models import load_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
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

from utils.data import load_credit_scoring_data
from utils.preprocessing import HighVIFDropper
from utils.entity_embedding import EntityEmbedder


def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


scorers = {
    "recall_score": make_scorer(recall_score),
    "f1_score": make_scorer(f1_score),
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score),
    "fbeta_score": make_scorer(f2),
    "auc": make_scorer(roc_auc_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}

key_results = [
    "mean_test_recall_score",
    "mean_test_f1_score",
    "mean_test_fbeta_score",
    "mean_test_accuracy_score",
    "mean_test_balanced_accuracy",
    "mean_test_auc",
    "mean_test_precision_score",
]


def main_logreg(data_path, descriptor_path, embedding_model, ds_name):
    # set up data
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    # set up preprocessing pipeline
    numeric_features = X.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer_num", SimpleImputer()),
            ("highVifDropper", "passthrough"),
            ("scaler", "passthrough"),
        ]
    )
    categorical_features = X.select_dtypes(include=("category", "bool")).columns
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoder", EntityEmbedder(embedding_model=embedding_model)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # set up logistic regression as a preprocessing + classifier pipeline
    logreg_pipe = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",)),
        ]
    )

    # set up grid search for preprocessing and classifier parameters
    params = {
        "clf__penalty": ["l1", "l2"],
        "clf__dual": [True, False],
        "clf__C": [130, 150, 160],
        "clf__solver": ["lbfgs", "liblinear"],
        "preprocessing__numerical__imputer_num__strategy": ["median", "mean"],
        "preprocessing__numerical__highVifDropper": [HighVIFDropper(), "passthrough"],
        "preprocessing__numerical__scaler": [RobustScaler(), StandardScaler()],
        "preprocessing__categorical__base_encoder": [
            OrdinalEncoder(categories=encoding_cats)
        ],
        "preprocessing__categorical__encoder": [
            EntityEmbedder(embedding_model=embedding_model),
            OneHotEncoder(drop="first"),
            "passthrough",
        ],
    }

    logreg_grid = GridSearchCV(
        logreg_pipe, param_grid=params, cv=3, scoring=scorers, refit="fbeta_score",
    )

    # fit pipeline to training data
    logreg_grid.fit(X_train, y_train)

    # get score and classification report for test data
    preds = logreg_grid.predict(X_test)

    # get best score and parameters and classification report for training data
    with open(f"logreg_results_{ds_name}.txt", "w") as f:
        f.write(f"Best fbeta_score on train set: {logreg_grid.best_score_:.3f}\n")
        f.write(f"Best parameter set: {logreg_grid.best_params_}\n")
        f.write(f"Best scores index: {logreg_grid.best_index_}\n")
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, logreg_grid.predict(X_train))}"
        )

        f.write(f"Scores for test set: {classification_report(y_test, preds)}\n")
        f.write(f"f1 score on test set: {f1_score(y_test, preds):.3f}")
        f.write(f"fbeta_score on test set: {f2(y_test, preds):.3f}")

    pd.DataFrame(logreg_grid.cv_results_)[key_results].to_csv(
        f"{ds_name}_logreg_cv_results.csv"
    )


if __name__ == "__main__":
    from pathlib import Path

    for ds_name in ["bene1"]:
        print(ds_name)
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_logreg(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
