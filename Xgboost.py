import numpy as np

from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
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
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    balanced_accuracy_score,
)

from utils.data import load_credit_scoring_data
from utils.preprocessing import HighVIFDropper
from utils.entity_embedding import EntityEmbedder
from utils.sklearn_results_plotting import (
    evaluate_metrics,
    plot_roc,
    plot_cm,
    best_model_parameters,
)


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


def main_xgboost(data_path, descriptor_path, embedding_model, ds_name):
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
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    # set up a base encoder to allow EntityEmbedder to receive numeric values
    base_ordinal_encoder = OrdinalEncoder(categories=encoding_cats)
    encoded_X = base_ordinal_encoder.fit_transform(
        X.select_dtypes(include=["category", "bool"]).values
    )

    # define possible categories of encoded variables for one hot encoder
    post_encoding_cats = [np.unique(col) for col in encoded_X.T]

    categorical_pipeline = Pipeline(
        steps=[
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
    xgboost_pipe = Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "clf",
                XGBClassifier(
                    verbosity=0,
                    objective="binary:logistic",
                    max_delta_step=1,
                    eval_metric="auc",
                ),
            ),
        ]
    )

    # set up grid search for preprocessing options and classifier parameters
    params = {
        "clf__max_depth": [4],
        "clf__learning_rate": [0.05],
        "clf__subsample": [1],
        "clf__scale_pos_weight": [1],
        "clf__min_child_weight": [3],
        "clf__booster": ["gbtree"],
        "clf__reg_lambda": [1],
        "clf__gamma": [0.1],
        "clf__colsample_bytree": [0.8],
        "clf__n_estimators": [250, 300],
        "preprocessing__numerical__highVifDropper": [HighVIFDropper(), "passthrough"],
        "preprocessing__numerical__scaler": [
            StandardScaler(),
            RobustScaler(),
            "passthrough",
        ],
        "preprocessing__categorical__base_encoder": [
            OrdinalEncoder(categories=encoding_cats),
            "passthrough",
        ],
        "preprocessing__categorical__encoder": [
            EntityEmbedder(embedding_model=embedding_model),
            OneHotEncoder(categories=post_encoding_cats, drop="first"),
            OneHotEncoder(categories=post_encoding_cats),
            "passthrough",
        ],
    }
    #
    inner_cv = KFold(n_splits=4, shuffle=True)
    outer_cv = KFold(n_splits=4, shuffle=True)

    # define grid search for classifier
    xgboost_grid = GridSearchCV(
        xgboost_pipe,
        param_grid=params,
        cv=inner_cv,
        scoring=scorers,
        refit="auc",
        verbose=1,
    )

    # fit pipeline to training data
    xgboost_model = xgboost_grid.fit(X_train, y_train)

    # calculate nested validation scores
    nested_score = cross_val_score(xgboost_grid, X_train, y_train, cv=outer_cv)

    # generate predictions for test data using fitted model
    class_preds = xgboost_model.predict(X_test)
    proba_preds = xgboost_model.predict_proba(X_test)

    # save best model
    joblib.dump(xgboost_model.best_estimator_, f"models/xgboost_{ds_name}.pkl")

    # get best parameters and classification report for training data
    best_model_parameters(
        X_train,
        y_train,
        y_test,
        class_preds,
        nested_score=nested_score,
        clf_name="xgboost",
        model=xgboost_model,
        ds_name=ds_name,
    )
    # get evaluation metrics for test data
    evaluate_metrics(
        y_test, class_preds, proba_preds, clf_name="xgboost", ds_name=ds_name
    )
    # plot confusion matrix
    plot_cm(y_test, class_preds, modelname=f"xgboost_{ds_name}")
    # plot roc
    plot_roc(y_test, class_preds, proba_preds, modelname=f"xgboost_{ds_name}")


if __name__ == "__main__":
    from pathlib import Path

    # for each data set:
    for ds_name in ["bene2"]:
        print(ds_name)
        # define embedding model saved model file
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_xgboost(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
