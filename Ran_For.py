import numpy as np

from keras.models import load_model
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

# from sklearn.externals import joblib
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
    # RandomizedSearchCV,
    KFold,
    cross_validate,
)
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    balanced_accuracy_score, brier_score_loss,
)

from utils.data_loading import load_credit_scoring_data
from utils.highVIFdropper import HighVIFDropper
from utils.entity_embedding import EntityEmbedder
from utils.model_evaluation import (
    plot_roc,
    plot_cm,
    evaluate_sklearn, make_ks_plot,
)


# define fbeta metric with beta = 3
def f3(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)


# define scorers
scorers = {
    "auc": "roc_auc",
    "f1_score": make_scorer(f1_score),
    "fbeta_score": make_scorer(f3),
    "accuracy_score": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "precision_score": make_scorer(precision_score),
    "recall_score": make_scorer(recall_score),
    "brier_score_loss": make_scorer(brier_score_loss),

}


def main_ranfor(data_path, descriptor_path, embedding_model, ds_name):

    # load data
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    # set up preprocessing pipelines
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
    encoding_cats = [X[i].unique().tolist() for i in categorical_features]

    # set up a base encoder to allow EntityEmbedder to receive numeric values
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

    # define pipeline with an oversampler, preprocessor and classifier
    ranfor_pipe = Pipeline(
        [
            ("oversampler", RandomOverSampler(sampling_strategy=0.8, random_state=42)),
            ("preprocessing", preprocessor),
            ("clf", RandomForestClassifier(),),
        ]
    )

    # set up grid search for preprocessing options and classifier parameters
    params = {
        "clf__n_estimators": [500],
        "clf__max_depth": [30],
        "clf__min_samples_split": [2],
        "clf__min_samples_leaf": [2],
        "clf__criterion": ["gini"],
        "preprocessing__numerical__highVifDropper": [HighVIFDropper(), "passthrough"],
        "preprocessing__numerical__scaler": [
            RobustScaler(),
            StandardScaler(),
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

    # define nested cross validation parameters
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=7)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=13)

    # define grid search for classifier
    ranfor_grid = GridSearchCV(
        ranfor_pipe,
        param_grid=params,
        cv=inner_cv,
        scoring=scorers,
        refit="auc",
        verbose=1,
    )

    # fit pipeline to cross validated data
    ranfor_model = ranfor_grid.fit(X_train, y_train)

    # calculate nested validation scores
    scores = cross_validate(
        ranfor_model, X_train, y_train, cv=outer_cv, scoring=scorers
    )

    clf = "ranfor"

    # # generate predictions for test data using fitted model
    train_preds = ranfor_model.predict(X_train)
    class_preds = ranfor_model.predict(X_test)

    # get best parameters and CV metrics
    evaluate_sklearn(
        y_test,
        proba_preds=class_preds,
        scores=scores,
        clf_name=clf,
        model=ranfor_model,
        ds_name=ds_name,
    )

    plot_cm(y_test, class_preds, clf_name=clf, modelname=f"{clf}_{ds_name}", iter="")
    plot_roc(y_test, class_preds, clf_name=clf, modelname=f"{clf}_{ds_name}", iter="")

    """KS plot working correctly- see model evaluation for more info"""
    make_ks_plot(y_train, train_preds, y_test, class_preds, clf, modelname=f"{clf}_{ds_name}", iter="")

    # # save best model
    # joblib.dump(ranfor_model.best_estimator_,
    # f"[old]results_plots/models/{clf}_{ds_name}.pkl")


if __name__ == "__main__":
    from pathlib import Path

    # for each dataset:
    for ds_name in ["bene2"]:
        print(ds_name)
        # define embedding model saved model file
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_ranfor(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
