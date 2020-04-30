import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from kerastuner import RandomSearch, Objective
from tensorflow.keras.utils import plot_model
from imblearn.over_sampling import RandomOverSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from utils.data_loading import load_credit_scoring_data
from utils.highVIFdropper import HighVIFDropper
from utils.model_evaluation import evaluate_metrics, plot_cm, plot_roc, roc_iter


def preprocess(X, X_train, y_train, X_test):

    # define no. of bins
    n_bins = 10

    # implement oversampling on train set
    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # set up numeric pipeline with binning
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

    # apply numeric pipeline in column transformer
    preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )

    # fit preprocessing on train set, and apply to train and test
    X_train_binned = preprocessor.fit_transform(X_train, y_train)
    X_test_binned = preprocessor.transform(X_test)

    # define shape variables
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

    # reformat instances from lists into arrays
    instances_train = np.array(instances_train)
    instances_test = np.array(instances_test)

    # reshape train, test sets to be compatible with 2D CNN
    X_train_final = instances_train.reshape(n_inst_train, n_bins, n_var, 1)
    X_test_final = instances_test.reshape(n_inst_test, n_bins, n_var, 1)

    return X_train_final, y_train, X_test_final, n_var, n_bins


def buildmodel(hp):

    model = None

    # define hyperparameter choices
    filters = hp.Choice("filters", values=[4, 8, 16, 32])
    dense1 = hp.Choice("dense_1", values=[2, 4, 8])

    # set up keras model
    model = keras.Sequential()

    # add 2D CNN layer
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=2,
            activation="relu",
            input_shape=(n_bins, n_var, 1),
            padding="same",
        )
    )

    # add pooling layer with different strategy choices
    if hp.Choice("pooling_", values=["avg", "max"]) == "max":
        model.add(keras.layers.MaxPooling2D(pool_size=2))
    else:
        model.add(keras.layers.AveragePooling2D(pool_size=2))

    # flatten 2D output to feed into dense layer
    model.add(keras.layers.Flatten())

    # add final dense layers
    model.add(keras.layers.Dense(dense1, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # compile model, define hyperparameter choices for learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float(
                "learning_rate",
                min_value=1e-2,
                max_value=2e-1,
                sampling="LOG",
                default=1e-1,
            )
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def main_2Dcnn_base(data_path, descriptor_path, ds_name):
    global n_var
    global n_bins

    # load data
    X, y, _, _, _, _ = load_credit_scoring_data(
        data_path, descriptor_path, rearrange="True"
    )

    # define variables for results and plots
    clf = "2Dcnn_base"
    modelname = f"{clf}_{ds_name}"

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(10, 10))

    # set up 5-fold cross validation split
    n_split = 5
    for i, (train_index, test_index) in enumerate(
        StratifiedKFold(n_split, random_state=13, shuffle=True).split(X, y)
    ):
        iter = i + 1

        # split data into cross validation subsets based on split index
        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # apply all preprocessing within CV fold
        X_train, y_train, X_test, n_var, n_bins = preprocess(
            X, x_train_split, y_train, x_test_split
        )

        # set up random search parameters
        tuner = RandomSearch(
            hypermodel=buildmodel,
            objective=Objective("val_auc", direction="max"),
            max_trials=100,
            executions_per_trial=1,
            directory=f"kerastuner/{clf}",
            project_name=f"{ds_name}_tuning_{iter}",
            overwrite=True,
        )

        # train and tune model using random search
        tuner.search(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)
            ],
        )

        # define best model
        best_model = tuner.get_best_models(1)[0]

        # generate predictions for test data using best model
        proba_preds = best_model.predict(X_test)
        class_preds = best_model.predict_classes(X_test)

        # get CV metrics, plot CM, KS and ROC
        evaluate_metrics(y_test, class_preds, proba_preds, clf, ds_name, iter=iter)
        plot_cm(y_test, class_preds, clf, modelname=modelname, iter=iter, p=0.5)
        roc_iter(y_test, proba_preds, tprs, mean_fpr, aucs, iter)

    # plot model architecture
    plot_model(
        model=best_model,
        to_file=f"model_plots/{clf}_{ds_name}_model_plot.png",
        show_shapes=True,
    )

    # plot combined ROC for all CV folds
    plot_roc(tprs, aucs, mean_fpr, modelname=modelname)
    fig.savefig(f"results/{clf}/{modelname}_ROC.png")
    plt.close(fig)


if __name__ == "__main__":

    for ds_name in ["german", "UK", "bene1", "bene2"]:
        print(ds_name)

        # run main function
        main_2Dcnn_base(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )
