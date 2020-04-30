import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from keras.models import load_model

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from scikitplot.helpers import binary_ks_curve

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    balanced_accuracy_score,
    brier_score_loss,
)

from utils.data_loading import load_credit_scoring_data
from utils.highVIFdropper import HighVIFDropper
from utils.entity_embedding import EntityEmbedder
from utils.model_evaluation import (
    plot_roc,
    plot_cm,
    evaluate_metrics,
    plot_KS,
    evaluate_parameters,
    roc_iter,
)


# define fbeta metric with beta = 3
def f3(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)


# define ks statistic
def ks(y_true, y_pred):
    res = binary_ks_curve(y_true, y_pred)
    ks_stat = res[3]
    return ks_stat


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
    "ks_stat": make_scorer(ks),
}


def main_xgboost(data_path, descriptor_path, embedding_model, ds_name):

    # load data
    X, y, _, _, _, _ = load_credit_scoring_data(data_path, descriptor_path)

    # set up numeric pipeline template
    numeric_features = X.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer_num", SimpleImputer()),
            ("highVifDropper", "passthrough"),
            ("scaler", "passthrough"),
        ]
    )

    # set up categorical pipeline
    categorical_features = X.select_dtypes(include=("category", "bool")).columns

    # define the possible categories of each variable in the dataset
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    # set up a base encoder to allow EntityEmbedder/ OneHot to receive numeric values
    base_ordinal_encoder = OrdinalEncoder(categories=encoding_cats)

    # define possible categories of encoded variables for One-Hot encoder
    encoded_X = base_ordinal_encoder.fit_transform(
        X.select_dtypes(include=["category", "bool"]).values
    )
    post_encoding_cats = [np.unique(col) for col in encoded_X.T]

    # categorical pipeline template
    categorical_pipeline = Pipeline(
        steps=[
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoder", "passthrough"),
        ]
    )

    # combine pipelines in column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # set-up full pipeline with an oversampler, preprocessor and classifier
    xgboost_pipe = Pipeline(
        [
            ("oversampling", RandomOverSampler(sampling_strategy=0.8, random_state=42)),
            ("preprocessing", preprocessor),
            (
                "clf",
                XGBClassifier(
                    verbosity=0,
                    objective="binary:logistic",
                    max_delta_step=1,
                    subsample=1,
                    scale_pos_weight=1,
                    booster="gbtree",
                    eval_metric="auc",
                    reg_lambda=1,
                    reg_alpha=0,
                ),
            ),
        ]
    )

    # set up grid search for preprocessing options and classifier hyperparameters
    params = {
        "clf__max_depth": [3, 4, 5, 6, 7, 8, 9],
        "clf__learning_rate": stats.uniform(0.01, 0.6),
        "clf__min_child_weight": [1, 2, 3, 4, 5],
        "clf__gamma": stats.uniform(0, 0.5),
        "clf__colsample_bytree": stats.uniform(0.5, 0.5),
        "clf__n_estimators": stats.randint(150, 1000),
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
            OneHotEncoder(categories=post_encoding_cats),
            "passthrough",
        ],
    }

    # define nested cross validation parameters
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

    # define random search for classifier
    xgboost_grid = RandomizedSearchCV(
        xgboost_pipe,
        param_distributions=params,
        n_iter=100,
        cv=inner_cv,
        scoring=scorers,
        refit="auc",
        verbose=1,
    )

    # define variables for results and plot naming
    clf = "xgboost"
    modelname = f"{clf}_{ds_name}"

    # set up variables and plots for results
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(10, 10))

    # perform cross validation
    for i, (train_index, test_index) in enumerate((outer_cv).split(X, y)):
        iter = i + 1

        # split data into cross validation subsets based on split index
        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # apply random search fitting to train data
        xgboost_model = xgboost_grid.fit(x_train_split, y_train)

        # generate predictions for test data using fitted model
        class_preds = xgboost_model.predict(x_test_split)
        proba_preds = xgboost_model.predict_proba(x_test_split)[:, 1]
        ks_preds = xgboost_model.predict_proba(x_test_split)

        # get best parameters and CV metrics, plot CM, KS and ROC
        evaluate_parameters(
            clf_name=clf, model=xgboost_model, ds_name=ds_name, iter=iter,
        )
        evaluate_metrics(y_test, class_preds, proba_preds, clf, ds_name, iter=iter)
        plot_cm(y_test, class_preds, clf_name=clf, modelname=modelname, iter=iter)
        plot_KS(y_test, ks_preds, clf_name=clf, modelname=modelname, iter=iter)
        roc_iter(y_test, proba_preds, tprs, mean_fpr, aucs, iter)

    # plot combined ROC for all CV folds
    plot_roc(tprs, aucs, mean_fpr, modelname=modelname)
    fig.savefig(f"results/{clf}/{modelname}_ROC.png")
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    # for each data set:
    for ds_name in ["german", "UK", "bene1", "bene2"]:
        print(ds_name)
        # define embedding model from saved model file
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        # run main function
        main_xgboost(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
