from sklearn.model_selection import KFold
import keras.backend as K
import tensorflow as tf

import keras
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from imblearn.over_sampling import RandomOverSampler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

from utils.data_loading import load_credit_scoring_data
from utils.entity_embedding import EntityEmbedder
from old_scripts.draft_scripts.preprocessing import HighVIFDropper


# define metrics
METRICS = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

np.random.seed = 50

early_stopping_auc = keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=10, mode="max", restore_best_weights=True
)


def preprocess(X, X_train, y_train, X_test, embedding_model):

    n_bins = 10
    oversampler = RandomOverSampler(sampling_strategy=0.8)
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

    # shuffle columns of instance matrices to change spatial relationships
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
def create_model(n_bins, n_var):
    model = Sequential()  # add model layers
    model.add(
        Conv2D(
            filters=6,
            kernel_size=(4, 8),
            padding="same",
            activation="relu",
            input_shape=(n_bins, n_var, 1),
            kernel_initializer="normal",
        )
    )
    model.add(Dropout(0.2))
    model.add(MaxPooling2D()),
    # model.add(Conv2D(filters=10, kernel_size=(3,4), activation="relu"))
    # model.add(MaxPooling2D()),
    # model.add(Dropout(0.2)),
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    adam = keras.optimizers.Adam(
        learning_rate=0.05, beta_1=0.95, beta_2=0.999, amsgrad=True
    )

    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=METRICS)

    return model


def train_evaluate_model(model, X_train_final, y_train, X_test_final, y_test):
    model.fit(
        X_train_final,
        y_train,
        epochs=100,
        # validation_data=(X_test_final, y_test),
        # callbacks=[early_stopping_auc],
    )

    return model.evaluate(X_test_final, y_test)


# main cnn function
def main_cnn_trans(data_path, descriptor_path, embedding_model, ds_name):

    # load and split data
    X, y, _, _, _, _, = load_credit_scoring_data(
        data_path, descriptor_path, rearrange="Emb"
    )

    n_split = 3
    aucscores = []

    for train_index, test_index in KFold(n_split).split(X):
        x_train_split, x_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_final, y_train, X_test_final, n_var, n_bins = preprocess(
            X, x_train_split, y_train, x_test_split, embedding_model
        )

        model = None
        model = Sequential()  # add model layers
        model.add(
            Conv2D(
                filters=6,
                kernel_size=(4, 8),
                padding="same",
                activation="relu",
                input_shape=(n_bins, n_var, 1),
                kernel_initializer="normal",
            )
        )
        model.add(Dropout(0.2))
        model.add(MaxPooling2D()),
        # model.add(Conv2D(filters=10, kernel_size=(3,4), activation="relu"))
        # model.add(MaxPooling2D()),
        # model.add(Dropout(0.2)),
        model.add(Flatten())
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        adam = keras.optimizers.Adam(
            learning_rate=0.05, beta_1=0.95, beta_2=0.999, amsgrad=True
        )

        model.compile(optimizer=adam, loss="binary_crossentropy", metrics=METRICS)

        model.fit(
            X_train_final,
            y_train,
            epochs=10,
            validation_data=(X_test_final, y_test),
            callbacks=[early_stopping_auc],
        )

        scores = model.evaluate(X_test_final, y_test)

        print(model.metrics_names)
        print(scores)
        aucscores.append(scores[-1])
        K.clear_session()
    print("%.2f%% (+/- %.2f%%)" % (np.mean(aucscores), np.std(aucscores)))

    # model.save(f"models/cnn_emb_{ds_name}.h5")
    # model.save_weights(f"models/weights/cnn_emb_weights_{ds_name}.h5")


if __name__ == "__main__":
    from pathlib import Path

    for ds_name in ["bene2"]:
        print(ds_name)
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_cnn_trans(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
