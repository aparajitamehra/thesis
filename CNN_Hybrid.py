import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from kerastuner import Objective, RandomSearch
from keras.engine.saving import load_model
from scipy.sparse import isspmatrix

from imblearn.over_sampling import RandomOverSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold

from utils.data_loading import load_credit_scoring_data
from utils.entity_embedding import EntityEmbedder
from utils.highVIFdropper import HighVIFDropper
from utils.model_evaluation import evaluate_metrics, plot_cm, plot_roc, roc_iter


def preprocess(X, X_train, y_train, X_test, y_test, embedding_model):

    # define no. of bins
    n_bins = 10

    # implement oversampling on train set
    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # set up numeric pipeline with binning for CNN input
    numeric_features = X.select_dtypes("number").columns

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

    # apply pipeline in column transformer
    num_preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )

    # fit numerical preprocessing on train set, and apply to train and test
    X_train_num = num_preprocessor.fit_transform(X_train, y_train)
    X_test_num = num_preprocessor.transform(X_test)

    # define shape variables
    n_inst_train = X_train_num.shape[0]
    n_inst_test = X_test_num.shape[0]
    n_num = int(X_train_num.shape[1] / n_bins)

    # add a n_bin x n_var matrix for each train instance to list
    instances_train = []
    for i in range(0, n_inst_train):
        row = X_train_num.getrow(i)
        row_reshaped = row.reshape(n_bins, n_num, order="F")
        row_dense = row_reshaped.todense()
        instances_train.append(row_dense)

    # add a n_bin x n_var matrix for each test instance to list
    instances_test = []
    for i in range(0, n_inst_test):
        row = X_test_num.getrow(i)
        row_reshaped = row.reshape(n_bins, n_num, order="F")
        row_dense = row_reshaped.todense()
        instances_test.append(row_dense)

    # reformat instances from lists into arrays
    instances_train = np.array(instances_train)
    instances_test = np.array(instances_test)

    # reshape train, test sets to be compatible with 2D CNN
    X_train_num = instances_train.reshape(n_inst_train, n_bins, n_num, 1)
    X_test_num = instances_test.reshape(n_inst_test, n_bins, n_num, 1)

    # set up categorical pipeline for ANN input
    categorical_features = X.select_dtypes(include=("bool", "category")).columns
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("baseencoder", OrdinalEncoder(categories=encoding_cats)),
            ("onehot", EntityEmbedder(embedding_model=embedding_model)),
        ]
    )
    # apply pipeline in column transformer
    cat_preprocessor = ColumnTransformer(
        transformers=[("categorical", categorical_pipeline, categorical_features)]
    )

    # fit categorical preprocessing on train set, and apply to train and test
    X_train_cat = cat_preprocessor.fit_transform(X_train, y_train)
    X_test_cat = cat_preprocessor.transform(X_test)

    # reformat data from sparse to dense matrix
    if isspmatrix(X_train_cat):
        X_train_cat = X_train_cat.todense()
    if isspmatrix(X_test_cat):
        X_test_cat = X_test_cat.todense()

    # define shape variables
    n_cat = X_train_cat.shape[-1]

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return (
        X_train_num,
        X_train_cat,
        y_train,
        X_test_num,
        X_test_cat,
        y_test,
        n_num,
        n_cat,
        n_bins,
    )


def buildmodel(hp):

    model = None

    """Numerical Input for CNN"""
    # define numerical input shape
    num_input = tf.keras.Input(shape=(n_bins, n_num, 1))

    # define hyperparameter choices
    filters = hp.Choice("filters", values=[4, 8, 16, 32])
    # kernel_size = hp.Choice('kernel_size', [(1,1),(3,3),(5,5)])
    dense = hp.Choice("dense", values=[2, 4, 8])

    # set up 2D CNN layer for numeric input
    conv11 = tf.keras.layers.Convolution2D(
        filters, kernel_size=(2, 2), activation="relu"
    )(num_input)

    # add pooling layer with different strategy choices
    if hp.Choice("pooling_", values=["avg", "max"]) == "max":
        pool11 = tf.keras.layers.MaxPool2D()(conv11)
    else:
        pool11 = keras.layers.AvgPool2D()(conv11)

    # flatten 2D output to feed into dense layer
    flat1 = keras.layers.Flatten()(pool11)

    """Categorical Input for ANN"""

    # define numerical input shape
    cat_input = tf.keras.Input(shape=(n_cat,))

    # set up dense ANN layer for categorical input
    dense21 = tf.keras.layers.Dense(dense, activation="relu")(cat_input)

    # merge CNN and ANN layers
    merge = tf.keras.layers.concatenate([flat1, dense21])

    # add final dense layers
    hidden1 = tf.keras.layers.Dense(dense, activation="relu")(merge)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden1)

    # set up full keras API model
    model = tf.keras.Model(inputs=[num_input, cat_input], outputs=output)

    # compile model, define hyperparameter choices for learning rate
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(
            hp.Float(
                "learning_rate",
                min_value=1e-2,
                max_value=2e-1,
                sampling="LOG",
                default=1e-1,
            )
        ),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def main_2Dcnn_hybrid(data_path, descriptor_path, ds_name):
    global n_num
    global n_cat
    global n_bins

    # load data
    X, y, _, _, _, _ = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True
    )

    # define variables for results and plots
    clf = "2Dcnn_hybrid"
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
        (
            X_train_num,
            X_train_cat,
            y_train,
            X_test_num,
            X_test_cat,
            y_test,
            n_num,
            n_cat,
            n_bins,
        ) = preprocess(X, x_train_split, y_train, x_test_split, y_test, embedding_model)

        # set up random search parameters
        tuner = RandomSearch(
            buildmodel,
            objective=Objective("val_auc", direction="max"),
            max_trials=100,
            executions_per_trial=1,
            directory=f"kerastuner/{clf}",
            project_name=f"{ds_name}_tuning_{iter}",
            overwrite=True,
        )

        # train and tune model using random search
        tuner.search(
            [X_train_num, X_train_cat],
            y_train,
            validation_data=([X_test_num, X_test_cat], y_test),
            epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)
            ],
        )

        # define best model
        best_model = tuner.get_best_models(1)[0]

        # generate predictions for test data using best model
        proba_preds = best_model.predict([X_test_num, X_test_cat])
        proba_preds = proba_preds[:, 0]
        class_preds = proba_preds.round()

        # get CV metrics, plot CM, KS and ROC
        evaluate_metrics(
            y_test, class_preds, proba_preds, proba_preds, clf, ds_name, iter=iter
        )
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
    from pathlib import Path

    for ds_name in ["german"]:
        print(ds_name)
        # define embedding model from saved model file
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        # run main function
        main_2Dcnn_hybrid(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )
