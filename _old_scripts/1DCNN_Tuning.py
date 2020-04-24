from __future__ import print_function

from utils.data_loading import load_credit_scoring_data

import tensorflow as tf
from kerastuner import Objective
from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model

from tensorflow import keras

from kerastuner.tuners import RandomSearch

from old_scripts.kerasformain import plot_cm, evaluate_metrics, prepcnn
import time

LOG_DIR = f"{int(time.time())}"


def build_tuned_cnn(hp):

    filters = hp.Choice("filters", values=[4, 8, 16, 32])
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
    """
    for i in range(hp.Int('nr_layers', 1, 2)):
        model.add(keras.layers.Conv1D(filters = filters, kernel_size=2, activation='relu', input_shape=(None,1), padding='same', strides=1))

    """
    """
    if hp.Choice('pooling_', values=['avg', 'max']) == 'max':
        model.add(keras.layers.MaxPooling1D(pool_size=2))
    else:
        model.add(keras.layers.AveragePooling1D(pool_size=2))
    """

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


def main(data_path, descriptor_path, embedding_model, ds_name):

    clf = "1D CNN"

    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True
    )

    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    X_train, y_train, X_test, y_test, X_val, y_val = prepcnn(
        X_train, X_test, y_train, y_test
    )

    tuner = RandomSearch(
        build_tuned_cnn,
        objective=Objective("val_auc", direction="max"),
        max_trials=100,
        executions_per_trial=2,
        directory="results_plots/{}".format(clf),
        project_name="{}_tuning".format(ds_name),
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)],
    )

    best_model = tuner.get_best_models(1)[0]

    model_json = best_model.to_json()
    with open("results_plots/{}/{}_model.json".format(clf, ds_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    best_model.save_weights("results_plots/{}/{}_model.h5".format(clf, ds_name))

    proba_preds_test = best_model.predict(X_test)
    class_preds_test = best_model.predict_classes(X_test)

    plot_cm(y_test, class_preds_test, modelname=clf, dsname=ds_name)
    plot_model(
        model=best_model,
        to_file="results_plots/{}/{}_model_plot.png".format(clf, ds_name),
        show_shapes=True,
    )

    evaluate_metrics(
        proba_preds_test, class_preds_test, y_test, clf_name=clf, ds_name=ds_name
    )


if __name__ == "__main__":
    from pathlib import Path

    for ds_name in ["UK", "bene1", "bene2", "german"]:
        print(ds_name)

        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )

    plt.show()
