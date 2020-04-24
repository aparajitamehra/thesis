import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.utils import plot_model
from keras.models import load_model
from kerastuner import Objective, RandomSearch

from imblearn.over_sampling import RandomOverSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

from utils.data_loading import load_credit_scoring_data
from utils.highVIFdropper import HighVIFDropper
from utils.entity_embedding import EntityEmbedder
from utils.model_evaluation import evaluate_keras, plot_cm, plot_roc


def preprocess(X, X_train, y_train, X_test, y_test, embedding_model):

    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    numeric_features = X.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("highVifDropper", HighVIFDropper()),
        ]
    )
    categorical_features = X.select_dtypes(include=("bool", "category")).columns
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoding", EntityEmbedder(embedding_model=embedding_model)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test


def buildmodel(hp):

    model = None
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(xdim,)))

    for i in range(hp.Int("nr_layers", 1, 3)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i + 1), min_value=2, max_value=16, step=2),
                activation="relu",
            )
        )

    model.add(keras.layers.Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def main_mlp(data_path, descriptor_path, embedding_model, ds_name):
    global xdim

    X, y, _, _, _, _ = load_credit_scoring_data(data_path, descriptor_path)

    clf = "mlp"
    n_split = 5
    aucscores = []

    for i, (train_index, test_index) in enumerate(StratifiedKFold(n_split, random_state=13, shuffle=True).split(X)):
        iter = i + 1

        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, y_train, X_test, y_test = preprocess(
            X, x_train_split, y_train, x_test_split, y_test, embedding_model
        )
        xdim = X_train.shape[-1]

        tuner = RandomSearch(
            buildmodel,
            objective=Objective("val_loss", direction="min"),
            max_trials=2,
            executions_per_trial=2,
            directory=f"kerastuner/{clf}",
            project_name=f"{ds_name}_tuning_{iter}",
            overwrite=True,
        )

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=2)],

        )
        best_model = tuner.get_best_models(1)[0]

        proba_preds = best_model.predict(X_test)
        class_preds = best_model.predict_classes(X_test)

        evaluate_keras(y_test, class_preds, proba_preds, clf, ds_name, iter=iter)
        plot_cm(
            y_test, class_preds, clf, modelname=f"{clf}_{ds_name}", iter=iter, p=0.5
        )
        plot_roc(y_test, proba_preds, clf, modelname=f"{clf}_{ds_name}", iter=iter)

        scores = best_model.evaluate(X_test, y_test)
        aucscores.append(scores[-1])
    print(aucscores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(aucscores), np.std(aucscores)))
    plot_model(
        model=best_model,
        to_file=f"model_plots/{clf}_{ds_name}_model_plot.png",
        show_shapes=True,
    )


if __name__ == "__main__":
    from pathlib import Path

    for ds_name in ["bene1"]:
        print(ds_name)

        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_mlp(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
