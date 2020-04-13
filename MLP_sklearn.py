import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


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
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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


# define fbeta metric with beta = 3
def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)


# define scorers
scorers = {
    "recall_score": make_scorer(recall_score),
    "f1_score": make_scorer(f1_score),
    "accuracy_score": make_scorer(accuracy_score),
    "precision_score": make_scorer(precision_score),
    "fbeta_score": make_scorer(f2),
    "auc": make_scorer(roc_auc_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}

# define key results to show in cv_results
key_results = [
    "mean_test_recall_score",
    "mean_test_f1_score",
    "mean_test_fbeta_score",
    "mean_test_accuracy_score",
    "mean_test_balanced_accuracy",
    "mean_test_auc",
    "mean_test_precision_score",
]


def main_mlp_sklearn(data_path, descriptor_path, embedding_model, ds_name):

    # load and split data
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    # oversample minority class to mitigate imbalance issue
    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # set up template preprocessing pipelines

    # numeric pipeline with HighVIF and Scaling
    numeric_features = X.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer_num", SimpleImputer()),
            ("highVifDropper", "passthrough"),
            ("scaler", "passthrough"),
        ]
    )

    # categorical pipeline with encoding options
    categorical_features = X.select_dtypes(include=("category", "bool")).columns

    # define the possible categories of each variable in the dataset
    encoding_cats = []
    for i in categorical_features:
        category = None
        try:
            category = [
                str(k) for k in sorted([int(j) for j in X[i].unique().tolist()])
            ]
        except ValueError:
            category = X[i].unique().tolist()
        encoding_cats.append(category)

    # set up a base encoder to allow Entity Embedder to receive numeric values
    base_ordinal_encoder = OrdinalEncoder(categories=encoding_cats)
    encoded_X = base_ordinal_encoder.fit_transform(
        X.select_dtypes(include=["category", "bool"]).values
    )

    # define possible categories of encoded variables for one hot encoder
    post_encoding_cats = [np.unique(col) for col in encoded_X.T]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoder", "passthrough"),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # define classifier as a preprocessor and classifier pipeline
    mlp_sklearn_pipe = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("clf", MLPClassifier(max_iter=5000, early_stopping=True)),
        ]
    )

    # set up grid search for preprocessing options and classifier parameters
    params = {
        "clf__batch_size": ["auto"],
        "clf__hidden_layer_sizes": [(64, 32, 1), (20, 10, 2)],
        "clf__activation": ["relu"],
        "clf__solver": ["adam"],
        # "clf__alpha": [0.001, 0.01],
        "clf__momentum": [0.8],
        "clf__learning_rate": ["adaptive"],
        "clf__learning_rate_init": [0.01, 0.1],
        "preprocessing__numerical__highVifDropper": [HighVIFDropper(), "passthrough"],
        "preprocessing__numerical__scaler": [RobustScaler(), StandardScaler()],
        "preprocessing__categorical__base_encoder": [
            OrdinalEncoder(categories=encoding_cats)
        ],
        "preprocessing__categorical__encoder": [
            EntityEmbedder(embedding_model=embedding_model),
            # OneHotEncoder(categories=post_encoding_cats, drop="first"),
            OneHotEncoder(categories=post_encoding_cats),
            "passthrough",
        ],
    }

    inner_cv = KFold(n_splits=4, shuffle=True)
    outer_cv = KFold(n_splits=4, shuffle=True)

    # define grid search for classifier
    mlp_sklearn_grid = GridSearchCV(
        mlp_sklearn_pipe, param_grid=params, cv=inner_cv, scoring=scorers, refit="auc",
    )

    # fit pipeline to training data
    mlp_sklearn_grid.fit(X_train, y_train)

    nested_score = cross_val_score(mlp_sklearn_grid, X_train, y_train, cv=outer_cv)

    # generate predictions for test data using fitted model
    preds = mlp_sklearn_grid.predict(X_test)

    joblib.dump(mlp_sklearn_grid.best_estimator_, f"models/mlp_sklearn_{ds_name}.pkl")

    # get best score and parameters and classification report for training data
    with open(f"mlp_sklearn_results_{ds_name}.txt", "w") as f:
        f.write(f"Best auc_score on train set: {mlp_sklearn_grid.best_score_:.3f}\n")
        f.write(f"Best parameter set: {mlp_sklearn_grid.best_params_}\n")
        f.write(f"Best scores index: {mlp_sklearn_grid.best_index_}\n")
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, mlp_sklearn_grid.predict(X_train))}"
        )

        f.write(f"Nested Scores: {nested_score.mean()}\n\n")

        # get score and classification report for test data
        f.write(f"Scores for test set: {classification_report(y_test, preds)}\n")
        f.write(f"f1 score on test set: {f1_score(y_test, preds):.3f}\n")
        f.write(f"fbeta_score on test set: {f2(y_test, preds):.3f}\n")
        f.write(f"AUC of test set: {roc_auc_score(y_test, preds):.3f}\n")
        f.write(f"Accuracy of test set: {accuracy_score(y_test, preds):.3f}\n")
        f.write(f"Precision of test set: {precision_score(y_test, preds):.3f}\n")
        f.write(f"Recall of test set: {recall_score(y_test, preds):.3f}\n")

    # write key cv results to csv file
    pd.DataFrame(mlp_sklearn_grid.cv_results_)[key_results].to_csv(
        f"mlp_sklearn_CV_results_{ds_name}.csv"
    )


if __name__ == "__main__":
    from pathlib import Path

    # for each data set:
    for ds_name in ["bene1"]:
        print(ds_name)
        # define embedding model saved model file
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

    main_mlp_sklearn(
        f"datasets/{ds_name}/input_{ds_name}.csv",
        f"datasets/{ds_name}/descriptor_{ds_name}.csv",
        embedding_model,
        ds_name,
    )
