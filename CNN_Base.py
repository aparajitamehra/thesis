import tensorflow as tf
from tensorflow import keras
import numpy as np

from kerastuner import RandomSearch, Objective
from tensorflow.keras.utils import plot_model
from imblearn.over_sampling import RandomOverSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from utils.data_loading import load_credit_scoring_data
from utils.highVIFdropper import HighVIFDropper
from utils.model_evaluation import evaluate_keras, plot_cm, plot_roc


def preprocess(X, X_train, y_train, X_test):
    n_bins = 10
    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    numeric_features = X.select_dtypes("number").columns
    print("n_numeric: ", numeric_features.size)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("highvif", HighVIFDropper()),
            (
                "binner",
                KBinsDiscretizer(n_bins=n_bins, strategy="uniform", encode="onehot"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )

    X_train_binned = preprocessor.fit_transform(X_train, y_train)
    X_test_binned = preprocessor.transform(X_test)

    n_inst_train = X_train_binned.shape[0]
    n_inst_test = X_test_binned.shape[0]
    n_var = int(X_train_binned.shape[1] / n_bins)

    # add a n_bin x n_var matrix for each train instance to list
    instances_train = []
    for i in range(0, n_inst_train):
        row = X_train_binned.getrow(i)
        row_reshaped = row.reshape(n_bins, n_var, order="F")
        row_dense = row_reshaped.todense()
        instances_train.append(row_dense)

    # add a n_bin x n_var matrix for each test instance to list
    instances_test = []
    for i in range(0, n_inst_test):
        row = X_test_binned.getrow(i)
        row_reshaped = row.reshape(n_bins, n_var, order="F")
        row_dense = row_reshaped.todense()
        instances_test.append(row_dense)

    # reformat instances from lists into arrays, so they can be shuffled in unison
    instances_train = np.array(instances_train)
    instances_test = np.array(instances_test)

    # reshape train, test and validation sets to make them appropriate input to CNN
    X_train_final = instances_train.reshape(n_inst_train, n_bins, n_var, 1)
    X_test_final = instances_test.reshape(n_inst_test, n_bins, n_var, 1)

    return X_train_final, y_train, X_test_final, n_var, n_bins


def buildmodel(hp):
    filters = hp.Choice("filters", values=[4, 8, 16, 32])
    # kernel = hp.Choice('kernel_size', values = [2,3])

    model = None
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=2,
            activation="relu",
            input_shape=(n_bins, n_var, 1),
            padding="same",
        )
    )

    if hp.Choice("pooling_", values=["avg", "max"]) == "max":
        model.add(keras.layers.MaxPooling2D(pool_size=2))
    else:
        model.add(keras.layers.AveragePooling2D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(8, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def main_2Dcnn_base(data_path, descriptor_path, ds_name):
    global n_var
    global n_bins

    X, y, _, _, _, _ = load_credit_scoring_data(
        data_path, descriptor_path, rearrange="True"
    )

    n_split = 3
    clf = "2Dcnn_base"
    aucscores = []

    for i, (train_index, test_index) in enumerate(KFold(n_split).split(X)):
        iter = i + 1

        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, y_train, X_test, n_var, n_bins = preprocess(
            X, x_train_split, y_train, x_test_split
        )

        tuner = RandomSearch(
            hypermodel=buildmodel,
            objective=Objective("val_auc", direction="max"),
            max_trials=3,
            # executions_per_trial=2,
            directory=f"kerastuner/{clf}",
            project_name=f"{ds_name}_tuning_{iter}",
            overwrite=True,
        )

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=3,
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

    for ds_name in ["bene2"]:
        print(ds_name)

        main_2Dcnn_base(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )