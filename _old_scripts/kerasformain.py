from __future__ import print_function

from tensorflow.keras.layers import Dense

import numpy as np
from old_scripts.draft_scripts.preprocessing import (
    MLPpreprocessing_pipeline_onehot,
    cnn2dprep_num,
    EntityPrep,
)  # CNNpreprocessing_pipeline_onehot
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import tempfile
from scipy.sparse import isspmatrix

from tensorflow.keras.utils import plot_model
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    fbeta_score,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def evaluate_metrics(proba_preds, class_preds, y_test, clf_name, ds_name):
    import csv

    with open(
        "results_plots/{}/{}metrics.csv".format(clf_name, clf_name), "a", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([ds_name, clf_name, "AUC", roc_auc_score(y_test, proba_preds)])
        writer.writerow([ds_name, clf_name, "F1 score", f1_score(y_test, class_preds)])
        writer.writerow(
            [ds_name, clf_name, "F beta", fbeta_score(y_test, class_preds, beta=3)]
        )
        writer.writerow(
            [ds_name, clf_name, "Accuracy", accuracy_score(y_test, class_preds)]
        )
        writer.writerow(
            [
                ds_name,
                clf_name,
                "Balanced Accuracy",
                balanced_accuracy_score(y_test, class_preds),
            ]
        )
        writer.writerow(
            [ds_name, clf_name, "Precision", precision_score(y_test, class_preds)]
        )
        writer.writerow(
            [ds_name, clf_name, "Recall", recall_score(y_test, class_preds)]
        )


# preprocessors


def prepmlp(X_train, X_test, y_train, y_test):

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2
    )

    # mlp preprocessing
    onehot_preprocessor = MLPpreprocessing_pipeline_onehot(X_train)
    onehot_preprocessor.fit(X_train)

    X_train = onehot_preprocessor.transform(X_train)
    X_test = onehot_preprocessor.transform(X_test)
    X_val = onehot_preprocessor.transform(X_val)

    if isspmatrix(X_train):
        X_train = X_train.todense()
    if isspmatrix(X_test):
        X_test = X_test.todense()
    if isspmatrix(X_val):
        X_val = X_val.todense()

    # min max capping
    X_train = np.clip(X_train, -5, 5)
    X_val = np.clip(X_val, -5, 5)
    X_test = np.clip(X_test, -5, 5)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val


def prep_ent_mlp(X_train, X_test, y_train, y_test):

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2
    )

    # mlp preprocessing
    entity_preprocessor = EntityPrep(X_train)
    entity_preprocessor.fit(X_train)

    X_train = entity_preprocessor.transform(X_train)
    X_test = entity_preprocessor.transform(X_test)
    X_val = entity_preprocessor.transform(X_val)

    if isspmatrix(X_train):
        X_train = X_train.todense()
    if isspmatrix(X_test):
        X_test = X_test.todense()
    if isspmatrix(X_val):
        X_val = X_val.todense()

    # min max capping
    X_train = np.clip(X_train, -5, 5)
    X_val = np.clip(X_val, -5, 5)
    X_test = np.clip(X_test, -5, 5)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val


def prepcnn(X_train, X_test, y_train, y_test):

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2
    )

    onehot_preprocessor = MLPpreprocessing_pipeline_onehot(X_train)
    onehot_preprocessor.fit(X_train)
    X_train = onehot_preprocessor.transform(X_train)
    X_test = onehot_preprocessor.transform(X_test)
    X_val = onehot_preprocessor.transform(X_val)

    print("type", type(X_train))

    if isspmatrix(X_train):
        X_train = X_train.toarray()
    if isspmatrix(X_test):
        X_test = X_test.toarray()
    if isspmatrix(X_val):
        X_val = X_val.toarray()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    y_val = y_val.to_numpy()

    print("pre func: ", X_train.shape)

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))

    y_train = y_train.reshape(y_train.shape[0],)
    y_test = y_test.reshape(y_test.shape[0],)
    y_val = y_val.reshape(y_val.shape[0],)

    print("post func: ", X_train.shape)
    return X_train, y_train, X_test, y_test, X_val, y_val


def prep2dcnn(X_train, X_test, y_train, y_test, binsize):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2, random_state=42
    )

    binned_prep = cnn2dprep_num(X_train, binsize)
    binned_prep.fit(X_train)
    X_train = binned_prep.transform(X_train)
    X_test = binned_prep.transform(X_test)
    X_val = binned_prep.transform(X_val)

    print("type", type(X_train))

    if isspmatrix(X_train):
        X_train = X_train.toarray()
    if isspmatrix(X_test):
        X_test = X_test.toarray()
    if isspmatrix(X_val):
        X_val = X_val.toarray()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    y_val = y_val.to_numpy()

    print("pre func: ", X_train.shape)

    ncols = int(X_train.shape[1] / binsize)
    print("ncols: ", ncols)
    X_train = X_train.reshape(X_train.shape[0], ncols, binsize, 1)
    X_test = X_test.reshape(X_test.shape[0], ncols, binsize, 1)
    X_val = X_val.reshape(X_val.shape[0], ncols, binsize, 1)

    y_train = y_train.reshape(y_train.shape[0],)
    y_test = y_test.reshape(y_test.shape[0],)
    y_val = y_val.reshape(y_val.shape[0],)

    print("post func: ", X_train.shape)
    return X_train, y_train, X_test, y_test, X_val, y_val, ncols


def prephybrid(X_train, X_test, y_train, y_test, binsize):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2, random_state=42
    )

    numeric_features = X_train.select_dtypes("number").columns
    categorical_features = X_train.select_dtypes(include=("bool", "category")).columns

    ##numerical
    X_train_num = X_train[numeric_features].copy()
    X_train_cat = X_train[categorical_features].copy()
    X_test_num = X_test[numeric_features].copy()
    X_test_cat = X_test[categorical_features].copy()
    X_val_num = X_val[numeric_features].copy()
    X_val_cat = X_val[categorical_features].copy()

    binned_prep = cnn2dprep_num(X_train_num, binsize)
    binned_prep.fit(X_train_num)

    X_train_num = binned_prep.transform(X_train_num)
    X_test_num = binned_prep.transform(X_test_num)
    X_val_num = binned_prep.transform(X_val_num)

    print("type", type(X_train_num))

    if isspmatrix(X_train_num):
        X_train_num = X_train_num.toarray()
    if isspmatrix(X_test_num):
        X_test_num = X_test_num.toarray()
    if isspmatrix(X_val_num):
        X_val_num = X_val_num.toarray()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    y_val = y_val.to_numpy()

    print("pre func: ", X_train.shape)

    n_num = int(X_train_num.shape[1] / binsize)
    print("n_num: ", n_num)
    X_train_num = X_train_num.reshape(X_train_num.shape[0], n_num, binsize, 1)
    X_test_num = X_test_num.reshape(X_test_num.shape[0], n_num, binsize, 1)
    X_val_num = X_val_num.reshape(X_val_num.shape[0], n_num, binsize, 1)

    print("post func: ", X_train_num.shape)

    ###categoricals

    onehot_preprocessor = MLPpreprocessing_pipeline_onehot(X_train_cat)
    onehot_preprocessor.fit(X_train_cat)

    X_train_cat = onehot_preprocessor.transform(X_train_cat)
    X_test_cat = onehot_preprocessor.transform(X_test_cat)
    X_val_cat = onehot_preprocessor.transform(X_val_cat)

    if isspmatrix(X_train_cat):
        X_train_cat = X_train_cat.todense()
    if isspmatrix(X_test_cat):
        X_test_cat = X_test_cat.todense()
    if isspmatrix(X_val_cat):
        X_val_cat = X_val_cat.todense()

    n_cat = X_train_cat.shape[-1]
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return (
        X_train_num,
        y_train,
        X_test_num,
        y_test,
        X_val_num,
        y_val,
        X_train_cat,
        X_test_cat,
        X_val_cat,
        n_num,
        n_cat,
    )


# model definitions
def make_2dcnn(
    metrics, binsize, nfeats, output_bias=None,
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    print("Build model...")
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=8,
                kernel_size=(2, 2),
                activation="relu",
                input_shape=(nfeats, binsize, 1),
                padding="same",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(lr=1e-2),
        metrics=metrics,
    )
    return model


def make_hybrid_model(
    metrics, binsize, n_num, n_cat, output_bias=None,
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    print("Build model...")

    # numerical
    num_input = Input(shape=(n_num, binsize, 1))
    conv11 = Conv2D(32, kernel_size=(2, 2), activation="relu")(num_input)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    # conv12 = Conv2D(16, kernel_size=(2,2), activation='relu')(pool11)
    # pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    # flat1 = Flatten()(pool12)
    flat1 = Flatten()(pool11)
    # categorical part
    cat_input = Input(shape=(n_cat,))
    dense21 = Dense(6, activation="relu")(cat_input)

    # merging
    merge = concatenate([flat1, dense21])
    hidden1 = Dense(10, activation="relu")(merge)
    output = Dense(1, activation="sigmoid")(hidden1)
    model = Model(inputs=[num_input, cat_input], outputs=output)

    print(model.summary())
    # plot graph
    plot_model(model, to_file="../old_outputs/multiple_inputs.png")

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model


def make_cnn(metrics, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    print("Build model...")
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
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(lr=1e-2),
        metrics=metrics,
    )
    return model


def make_MLP(xdim, metrics, output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    print("Build model...")
    model = keras.Sequential(
        [
            keras.layers.Dense(16, activation="relu", input_shape=(xdim,)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(lr=1e-2),
        metrics=metrics,
    )

    return model


def tuned_MLP_model(X_train, y_train, X_val, y_val, params):

    metrics = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]

    print("Build model...")
    model = keras.Sequential(
        [
            keras.layers.Dense(
                params["first_neuron"],
                activation=params["activation"],
                input_shape=(X_train.shape[-1],),
            ),
            keras.layers.Dropout(params["dropout"]),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=params["optimizer"](
            lr=lr_normalizer(params["lr"], params["optimizer"])
        ),
        metrics=metrics,
    )

    out = model.fit(
        x=X_train,
        y=y_train,
        validation_data=[X_val, y_val],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )

    # modify the output model
    return out, model


# plotting
def plot_cm(labels, predictions, modelname, dsname, p=0.5):

    fig = plt.figure()
    cm = confusion_matrix(labels, predictions > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix {}".format(modelname))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print("Legitimate Transactions Detected (True Negatives): ", cm[0][0])
    print("Legitimate Transactions Incorrectly Detected (False Positives): ", cm[0][1])
    print("Fraudulent Transactions Missed (False Negatives): ", cm[1][0])
    print("Fraudulent Transactions Detected (True Positives): ", cm[1][1])
    print("Total Fraudulent Transactions: ", np.sum(cm[1]))
    fig.savefig("results_plots/{}/{}_CM.png".format(modelname, dsname))
    plt.close(fig)


def plot_metrics(history, modtype, colors):

    mpl.rcParams["figure.figsize"] = (12, 10)
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    metrics = ["loss", "auc", "precision", "recall"]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(
            history.epoch,
            history.history[metric],
            color=colors,
            label=modtype + " Train",
        )
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors,
            linestyle="--",
            label=modtype + " Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.5, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


# model implementation
def makeweightedMLP(X_train, X_test, y_train, y_test, ds_name):

    print("makemlp")

    # X_train, y_train, X_test, y_test, X_val, y_val = prepmlp(X_train, X_test, y_train, y_test)
    X_train, y_train, X_test, y_test, X_val, y_val = prep_ent_mlp(
        X_train, X_test, y_train, y_test
    )
    xdim = X_train.shape[-1]
    EPOCHS = 100
    BATCH_SIZE = 2000

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
    # initial weights

    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", verbose=1, patience=10, mode="max", restore_best_weights=True
    )

    model = make_MLP(xdim=X_train.shape[-1], metrics=METRICS)
    model.summary()

    # set initial bias
    model = make_MLP(xdim, METRICS, output_bias=initial_bias)
    model.predict(X_train[:10])
    results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
    model.save_weights(initial_weights)

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print("Weight for class 0: {:.2f}".format(weight_for_0))
    print("Weight for class 1: {:.2f}".format(weight_for_1))

    weighted_model = make_MLP(xdim, METRICS)
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),
        class_weight=class_weight,
    )

    proba_preds_test = weighted_model.predict(X_test, batch_size=BATCH_SIZE)
    class_preds_test = weighted_model.predict_classes(X_test, batch_size=BATCH_SIZE)

    weighted_results = weighted_model.evaluate(
        X_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )

    print("Weighted model: ")
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ": ", value)
    print()

    plot_cm(y_test, class_preds_test, modelname="MLP")

    plot_model(
        model=weighted_model,
        to_file="../results_plots/keras_plots/MLP_model_plot.png",
        show_shapes=True,
    )

    evaluate_metrics(
        proba_preds_test, class_preds_test, y_test, clf_name="MLP", ds_name=ds_name
    )

    print("THESE ARE MLP PREDS")

    return weighted_history


def make_weighted2dCNN(X_train, X_test, y_train, y_test, ds_name):

    print("make2dcnn")
    print("pre: ", X_train.shape)
    xdim = X_train.shape[-1]
    binsize = 10
    print("Xdim: ", xdim)
    X_train, y_train, X_test, y_test, X_val, y_val, ncols = prep2dcnn(
        X_train, X_test, y_train, y_test, binsize=binsize
    )
    print("post: ", X_train.shape)
    EPOCHS = 100
    BATCH_SIZE = 2000

    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(),
    ]

    # initial weights
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", verbose=1, patience=10, mode="max", restore_best_weights=True
    )

    model = make_2dcnn(metrics=METRICS, binsize=binsize, nfeats=ncols)

    model.summary()

    # set initial bias

    model = make_2dcnn(METRICS, nfeats=ncols, binsize=binsize, output_bias=initial_bias)
    model.predict(X_train[:10])
    results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
    model.save_weights(initial_weights)

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print("Weight for class 0: {:.2f}".format(weight_for_0))
    print("Weight for class 1: {:.2f}".format(weight_for_1))

    weighted_model = make_2dcnn(METRICS, nfeats=ncols, binsize=binsize)
    weighted_model.summary()
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),
        class_weight=class_weight,
    )

    weighted_results = weighted_model.evaluate(
        X_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )

    print("Weighted model: ")
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ": ", value)
    print()

    proba_preds_test = weighted_model.predict(X_test)
    class_preds_test = weighted_model.predict_classes(X_test)

    plot_cm(y_test, class_preds_test, modelname="2dCNN")
    plot_model(
        model=weighted_model,
        to_file="../results_plots/keras_plots/2dCNN_model_plot.png",
        show_shapes=True,
    )

    evaluate_metrics(
        proba_preds_test, class_preds_test, y_test, clf_name="2D_CNN", ds_name=ds_name
    )

    return weighted_history


def make_weightedCNN(X_train, X_test, y_train, y_test, ds_name):
    print("makecnn")
    print("pre: ", X_train.shape)
    X_train, y_train, X_test, y_test, X_val, y_val = prepcnn(
        X_train, X_test, y_train, y_test
    )
    print("post: ", X_train.shape)
    EPOCHS = 100
    BATCH_SIZE = 2000

    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(),
    ]

    # initial weights

    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", verbose=1, patience=2, mode="max", restore_best_weights=True
    )

    model = make_cnn(metrics=METRICS)
    model.summary()

    # set initial bias
    model = make_cnn(METRICS, output_bias=initial_bias)
    model.predict(X_train[:10])
    results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
    model.save_weights(initial_weights)

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print("Weight for class 0: {:.2f}".format(weight_for_0))
    print("Weight for class 1: {:.2f}".format(weight_for_1))

    weighted_model = make_cnn(METRICS)
    weighted_model.summary()
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),
        class_weight=class_weight,
    )

    proba_preds_test = weighted_model.predict(X_test)
    class_preds_test = weighted_model.predict_classes(X_test)

    weighted_results = weighted_model.evaluate(
        X_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )

    print("Weighted model: ")
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ": ", value)
    print()

    plot_cm(y_test, class_preds_test, modelname="CNN")
    plot_model(
        model=weighted_model,
        to_file="../results_plots/keras_plots/CNN_model_plot.png",
        show_shapes=True,
    )

    evaluate_metrics(
        proba_preds_test, class_preds_test, y_test, clf_name="1D_CNN", ds_name=ds_name
    )
    return weighted_history


def make_weighted_hybrid_CNN(X_train, X_test, y_train, y_test, ds_name):

    print("make_hybrid_cnn")

    binsize = 10

    (
        X_train_num,
        y_train,
        X_test_num,
        y_test,
        X_val_num,
        y_val,
        X_train_cat,
        X_test_cat,
        X_val_cat,
        n_num,
        n_cat,
    ) = prephybrid(X_train, X_test, y_train, y_test, binsize=binsize)

    EPOCHS = 150
    BATCH_SIZE = 2000

    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(),
    ]

    # initial weights
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", verbose=1, patience=2, mode="max", restore_best_weights=True
    )

    model = make_hybrid_model(
        metrics=METRICS, binsize=binsize, n_cat=n_cat, n_num=n_num
    )

    model.summary()

    # set initial bias

    model = make_hybrid_model(
        metrics=METRICS,
        binsize=binsize,
        n_cat=n_cat,
        n_num=n_num,
        output_bias=initial_bias,
    )
    model.predict([X_train_num[:10], X_train_cat[:10]])
    results = model.evaluate(
        [X_train_num, X_train_cat], y_train, batch_size=BATCH_SIZE, verbose=0
    )
    initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
    model.save_weights(initial_weights)

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print("Weight for class 0: {:.2f}".format(weight_for_0))
    print("Weight for class 1: {:.2f}".format(weight_for_1))

    weighted_model = make_hybrid_model(
        metrics=METRICS, binsize=binsize, n_cat=n_cat, n_num=n_num
    )
    weighted_model.summary()
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        [X_train_num, X_train_cat],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=([X_val_num, X_val_cat], y_val),
        class_weight=class_weight,
    )

    weighted_results = weighted_model.evaluate(
        [X_test_num, X_test_cat], y_test, batch_size=BATCH_SIZE, verbose=0
    )
    print("Weighted model: ")
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ": ", value)

    proba_preds_test = weighted_model.predict([X_test_num, X_test_cat])

    proba_preds_test = proba_preds_test[:, 0]

    # class_preds_test = proba_preds_test.argmax(axis=-1)
    # temp hack
    class_preds_test = proba_preds_test.round()
    """
    print(proba_preds_test[:10])
    print("classes")
    print(class_preds_test[:10])
    """

    plot_cm(y_test, class_preds_test, modelname="hybrid")
    plot_model(
        model=weighted_model,
        to_file="../results_plots/keras_plots/Hybrid_model_plot.png",
        show_shapes=True,
    )

    evaluate_metrics(
        proba_preds_test, class_preds_test, y_test, clf_name="Hybrid", ds_name=ds_name
    )

    return weighted_history
