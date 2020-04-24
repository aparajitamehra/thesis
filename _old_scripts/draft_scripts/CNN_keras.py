import numpy as np
import keras
from keras.wrappers.scikit_learn import KerasClassifier

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
from sklearn.model_selection import train_test_split
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

from utils.data_loading import load_credit_scoring_data
from old_scripts.draft_scripts.preprocessing import HighVIFDropper
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

METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]


def main_cnn_keras(data_path, descriptor_path, embedding_model, ds_name):
    # set up data
    data = load_credit_scoring_data(data_path, descriptor_path)

    y = data.pop("censor")
    X = data

    # split train, test data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=20
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
            ("encoder", "passthrough"),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # set up CNN in keras as a preprocessing + classifier pipeline
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    def make_cnn(metrics=METRICS, output_bias=initial_bias):
        # if output_bias is not None:
        #     output_bias = tf.keras.initializers.Constant(output_bias)

        model = keras.Sequential(
            [
                keras.layers.Conv1D(
                    filters=8,
                    kernel_size=2,
                    activation="relu",
                    input_shape=(None, 1),
                    padding="same",
                    strides=1,
                ),
                keras.layers.GlobalMaxPooling1D(),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(lr=1e-2),
            metrics=metrics,
        )

        return model

    cnn_keras = KerasClassifier(build_fn=make_cnn, verbose=1)

    cnn_pipe = Pipeline([("preprocessing", preprocessor), ("clf", cnn_keras),])

    # set up grid search for preprocessing and classifier parameters
    params = {
        "preprocessing__numerical__imputer_num__strategy": ["median", "mean"],
        "preprocessing__numerical__highVifDropper": [HighVIFDropper(), "passthrough"],
        "preprocessing__numerical__scaler": [RobustScaler(), StandardScaler()],
        "preprocessing__categorical__base_encoder": [
            OrdinalEncoder(categories=encoding_cats)
        ],
        "preprocessing__categorical__encoder": [
            EntityEmbedder(embedding_model=embedding_model),
            OneHotEncoder(drop="first"),
            OneHotEncoder(),
            "passthrough",
        ],
    }

    cnn_keras_grid = GridSearchCV(
        cnn_pipe,
        param_grid=params,
        cv=3,
        scoring=scorers,
        refit="fbeta_score",
        n_jobs=-1,
    )

    # fit pipeline to training data
    cnn_keras_grid.fit(X_train, y_train)

    # get score and classification report for test data
    preds = cnn_keras_grid.predict(X_test)

    # get best score and parameters and classification report for training data
    with open(f"cnn_keras_results{ds_name}.txt", "w") as f:
        f.write(f"Best f1_score on train set: {cnn_keras_grid.best_score_:.3f}\n")
        f.write(f"Best parameter set: {cnn_keras_grid.best_params_}\n")
        f.write(f"Best scores index: {cnn_keras_grid.best_index_}\n")
        f.write(
            f"Scores for train set: {classification_report(y_train, cnn_keras_grid.predict(X_train))}"
        )

        f.write(f"Scores for test set: {classification_report(y_test, preds)}\n")
        f.write(f"f1 score on test set: {f1_score(y_test, preds):.3f}")


if __name__ == "__main__":
    from pathlib import Path

    for ds_name in ["german"]:
        print(ds_name)
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_cnn_keras(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
