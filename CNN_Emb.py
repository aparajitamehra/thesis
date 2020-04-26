import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from keras.models import load_model
from kerastuner import RandomSearch, Objective

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer

from utils.data_loading import load_credit_scoring_data
from utils.entity_embedding import EntityEmbedder
from utils.highVIFdropper import HighVIFDropper
from utils.model_evaluation import evaluate_metrics, plot_cm, plot_roc, roc_iter

np.random.seed = 50


def preprocess(X, X_train, y_train, X_test, embedding_model):

    n_bins = 10
    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    print(X_train.shape)
    print(y_train.shape)

    # transform categorical variables to embeddings, pass-through numerical
    categorical_features = X.select_dtypes(include=("category", "bool")).columns
    numeric_features = X.select_dtypes("number").columns
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer_num", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("highvif", HighVIFDropper()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoder", EntityEmbedder(embedding_model=embedding_model)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # fit train and test sets to preprocessor
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)

    # bin all variables into n_bins
    binning_pipeline = Pipeline(
        steps=[
            (
                "binner",
                KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="uniform"),
            )
        ]
    )

    # fit binning on train and test sets
    X_train_binned = binning_pipeline.fit_transform(X_train, y_train)
    X_test_binned = binning_pipeline.transform(X_test)

    # define shape variables
    n_inst_train = X_train_binned.shape[0]
    n_inst_test = X_test_binned.shape[0]
    n_var = X_train.shape[1]

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

    # randomly shuffle columns of instance matrices to change spatial relationships
    """
    all_instances = list(zip(instances_train.T, instances_val.T, instances_test.T))
    np.random.shuffle(all_instances)
    instances_train, instances_val, instances_test = zip(*all_instances)
    """
    """
    # reformat instances back to arrays after shuffle
    instances_train = np.array(instances_train).T
    instances_val = np.array(instances_val).T
    instances_test = np.array(instances_test).T
    """

    # reshape train, test and validation sets to make them appropriate input to CNN
    X_train_final = instances_train.reshape(n_inst_train, n_bins, n_var, 1)
    X_test_final = instances_test.reshape(n_inst_test, n_bins, n_var, 1)

    return X_train_final, y_train, X_test_final, n_var, n_bins


# build CNN
def buildmodel(hp):
    filters = hp.Choice("filters", values=[4, 8, 16, 32])
    hidden1 = hp.Choice("hidden1", values=[2,4,8])
    # kernel = hp.Choice('kernel_size', values = [2,3])

    model = None
    model = keras.Sequential()  # add model layers
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=2,
            padding="same",
            activation="relu",
            input_shape=(n_bins, n_var, 1),
            kernel_initializer="normal",
        )
    )
    model.add(keras.layers.Dropout(0.2))
    if hp.Choice("pooling_", values=["avg", "max"]) == "max":
        model.add(keras.layers.MaxPooling2D(pool_size=2))
    else:
        model.add(keras.layers.AveragePooling2D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(8, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    adam = keras.optimizers.Adam(
        hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
    )

    model.compile(
        optimizer=adam,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


# main cnn function
def main_2Dcnn_emb(data_path, descriptor_path, embedding_model, ds_name):
    global n_var
    global n_bins

    # load and split data
    X, y, _, _, _, _ = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True
    )

    n_split = 5
    clf = "2Dcnn_emb"
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

        X_train_final, y_train, X_test_final, n_var, n_bins = preprocess(
            X, x_train_split, y_train, x_test_split, embedding_model
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
            X_train_final,
            y_train,
            validation_data=(X_test_final, y_test),
            epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)
            ],
        )

        best_model = tuner.get_best_models(1)[0]

        proba_preds = best_model.predict(X_test_final)
        class_preds = best_model.predict_classes(X_test_final)

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
    from pathlib import Path

    for ds_name in ["bene2"]:
        print(ds_name)
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_2Dcnn_emb(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
