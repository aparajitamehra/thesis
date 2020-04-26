import tensorflow as tf
from scipy.sparse import isspmatrix
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from kerastuner import Objective, RandomSearch

from imblearn.over_sampling import RandomOverSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from utils.data_loading import load_credit_scoring_data
from utils.highVIFdropper import HighVIFDropper
from utils.model_evaluation import evaluate_metrics, plot_cm, plot_roc, roc_iter


def preprocess(X, X_train, y_train, X_test, y_test):

    n_bins = 10
    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

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
    num_preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_pipeline, numeric_features)]
    )

    X_train_num = num_preprocessor.fit_transform(X_train, y_train)
    X_test_num = num_preprocessor.transform(X_test)

    n_inst_train = X_train_num.shape[0]
    n_inst_test = X_test_num.shape[0]
    n_num = int(X_train_num.shape[1] / n_bins)

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

    # reformat instances from lists into arrays, so they can be shuffled in unison
    instances_train = np.array(instances_train)
    instances_test = np.array(instances_test)

    # reshape train, test and validation sets to make them appropriate input to CNN
    X_train_num = instances_train.reshape(n_inst_train, n_bins, n_num, 1)
    X_test_num = instances_test.reshape(n_inst_test, n_bins, n_num, 1)

    categorical_features = X.select_dtypes(include=("bool", "category")).columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    cat_preprocessor = ColumnTransformer(
        transformers=[("categorical", categorical_pipeline, categorical_features)]
    )

    cat_preprocessor.fit(X_train)
    X_train_cat = cat_preprocessor.transform(X_train)
    X_test_cat = cat_preprocessor.transform(X_test)

    if isspmatrix(X_train_cat):
        X_train_cat = X_train_cat.todense()
    if isspmatrix(X_test_cat):
        X_test_cat = X_test_cat.todense()

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

    # numerical
    num_input = tf.keras.Input(shape=(n_bins, n_num, 1))
    # x = num_input
    filters = hp.Choice("filters", values=[4, 8, 16, 32])
    cat_neurons=hp.Choice("cat_neurons", values =[2,4,8])
    hidden1_neurons=hp.Choice("hidden1_neurons", values =[2,4,8])
    # kernel_size = hp.Choice('kernel_size', [(1,1),(3,3),(5,5)])

    conv11 = tf.keras.layers.Convolution2D(
        filters, kernel_size=(2, 2), activation="relu"
    )(num_input)

    if hp.Choice("pooling_", values=["avg", "max"]) == "max":
        pool11 = tf.keras.layers.MaxPool2D()(conv11)
    else:
        pool11 = keras.layers.AvgPool2D()(conv11)

    # pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)

    # conv12 = Conv2D(16, kernel_size=(2,2), activation='relu')(pool11)
    # pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    # flat1 = Flatten()(pool12)

    flat1 = keras.layers.Flatten()(pool11)

    # categorical part
    cat_input = tf.keras.Input(shape=(n_cat,))

    dense21 = tf.keras.layers.Dense(cat_neurons, activation="relu")(cat_input)

    # merging
    merge = tf.keras.layers.concatenate([flat1, dense21])
    hidden1 = tf.keras.layers.Dense(hidden1_neurons, activation="relu")(merge)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden1)

    model = tf.keras.Model(inputs=[num_input, cat_input], outputs=output)

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
        ),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    """
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(hp.Float
                  ('learning_rate', 1e-4, 1e-2, sampling='log')),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    """
    return model


def main_2Dcnn_hybrid(data_path, descriptor_path, ds_name):
    global n_num
    global n_cat
    global n_bins

    X, y, _, _, _, _ = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True
    )

    n_split = 5
    clf = "2Dcnn_hybrid"
    modelname = f"{clf}_{ds_name}"

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(10, 10))

    for i, (train_index, test_index) in enumerate(
        StratifiedKFold(n_split, random_state=13, shuffle=True).split(X, y)
    ):
        iter = i + 1

        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
        ) = preprocess(X, x_train_split, y_train, x_test_split, y_test)

        tuner = RandomSearch(
            buildmodel,
            objective=Objective("val_auc", direction="max"),
            max_trials=100,
            executions_per_trial=1,
            directory=f"kerastuner/{clf}",
            project_name=f"{ds_name}_tuning_{iter}",
            overwrite=True,
        )

        tuner.search(
            [X_train_num, X_train_cat],
            y_train,
            validation_data=([X_test_num, X_test_cat], y_test),
            epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)
            ],
        )

        best_model = tuner.get_best_models(1)[0]

        proba_preds = best_model.predict([X_test_num, X_test_cat])
        proba_preds = proba_preds[:, 0]
        class_preds = proba_preds.round()

        evaluate_metrics(y_test, class_preds, proba_preds, clf, ds_name, iter=iter)
        plot_cm(y_test, class_preds, clf, modelname=modelname, iter=iter, p=0.5)
        roc_iter(y_test, proba_preds, tprs, mean_fpr, aucs, iter)

    plot_model(
        model=best_model,
        to_file=f"model_plots/{clf}_{ds_name}_model_plot.png",
        show_shapes=True,
    )
    plot_roc(tprs, aucs, mean_fpr, clf, modelname=modelname)
    fig.savefig(f"results/{clf}/{modelname}_ROC.png")
    plt.close(fig)


if __name__ == "__main__":

    for ds_name in ["UK","bene2","bene1","german"]:
        print(ds_name)

        main_2Dcnn_hybrid(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            ds_name,
        )
