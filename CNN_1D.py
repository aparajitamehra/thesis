import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.utils import plot_model
from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model
from kerastuner import RandomSearch, Objective

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold


from utils.data_loading import load_credit_scoring_data
from utils.entity_embedding import EntityEmbedder
from utils.highVIFdropper import HighVIFDropper
from utils.model_evaluation import evaluate_keras, plot_cm, plot_roc


def preprocess(X, X_train, y_train, X_test, y_test, embedding_model):

    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    numeric_features = X.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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

    print(X_train.shape)

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    y_train = y_train.reshape(y_train.shape[0],)
    y_test = y_test.reshape(y_test.shape[0],)

    return X_train, y_train, X_test, y_test


def buildmodel(hp):

    filters = hp.Choice("filters", values=[4, 8, 16, 32])

    model = None
    model = keras.Sequential()

    model.add(
        keras.layers.Conv1D(
            filters=filters,
            kernel_size=2,
            activation="relu",
            input_shape=(None, 1),
            padding="same",
            strides=1,
        )
    )
    model.add(keras.layers.GlobalMaxPooling1D())

    # for i in range(hp.Int('nr_layers', 1, 2)):
    #     model.add(keras.layers.Conv1D(filters=filters,
    #     kernel_size=2,
    #     activation='relu',
    #     input_shape=(None,1),
    #     padding='same', strides=1))

    # if hp.Choice('pooling_', values=['avg', 'max']) == 'max':
    #     model.add(keras.layers.MaxPooling1D(pool_size=2))
    # else:
    #     model.add(keras.layers.AveragePooling1D(pool_size=2))

    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main_1Dcnn(data_path, descriptor_path, embedding_model, ds_name):

    X, y, _, _, _, _ = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True
    )

    n_split = 5
    clf = "1Dcnn"
    aucscores = []

    for i, (train_index, test_index) in enumerate(KFold(n_split, random_state=13).split(X)):
        iter = i + 1

        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, y_train, X_test, y_test = preprocess(
            X, x_train_split, y_train, x_test_split, y_test, embedding_model
        )

        tuner = RandomSearch(
            hypermodel=buildmodel,
            objective=Objective("val_auc", direction="max"),
            max_trials=100,
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
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)],
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

        main_1Dcnn(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
