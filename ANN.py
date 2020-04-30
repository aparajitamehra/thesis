import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from tensorflow.keras.utils import plot_model
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
from utils.model_evaluation import evaluate_metrics, plot_cm, plot_roc, roc_iter


def preprocess(X, X_train, y_train, X_test, y_test, embedding_model):

    # implement oversampling on train set
    oversampler = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # set up numeric pipeline
    numeric_features = X.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("highVifDropper", HighVIFDropper()),
        ]
    )

    # set up categorical pipeline
    categorical_features = X.select_dtypes(include=("bool", "category")).columns

    # define the possible categories of each variable in the dataset
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoding", EntityEmbedder(embedding_model=embedding_model)),
        ]
    )

    # combine pipelines in column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # fit preprocessing on train set, and apply to train and test
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test


def buildmodel(hp):

    model = None

    # set up keras ANN
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(xdim,)))

    # define hyperparameter choices for no. layers/ no.units
    for i in range(hp.Int("nr_layers", 2, 4)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i + 1), min_value=2, max_value=16, step=2),
                activation="relu",
            )
        )

    # add final dense layer
    model.add(keras.layers.Dense(1, activation="sigmoid", name="output"))

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


def main_ann(data_path, descriptor_path, embedding_model, ds_name):
    global xdim

    # load data
    X, y, _, _, _, _ = load_credit_scoring_data(data_path, descriptor_path)

    # define variables for results and plots
    clf = "ann"
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
        X_train, y_train, X_test, y_test = preprocess(
            X, x_train_split, y_train, x_test_split, y_test, embedding_model
        )
        xdim = X_train.shape[-1]

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
    from pathlib import Path

    for ds_name in ["german", "UK", "bene1", "bene2"]:
        print(ds_name)
        # define embedding model from saved model file
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        # run main function
        main_ann(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
