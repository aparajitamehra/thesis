import numpy as np
from sklearn.metrics import classification_report

from utils.data import load_credit_scoring_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from utils.entity_embedding import EntityEmbedder
from keras.models import load_model
from sklearn.preprocessing import KBinsDiscretizer

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


def main_cnn_trans(data_path, descriptor_path, embedding_model, ds_name):
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )
    n_bins = 8

    # preprocess variables- transform categorical variables to embeddings
    categorical_features = X.select_dtypes(include=("category", "bool")).columns
    numeric_features = X.select_dtypes("number").columns
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
            ("base_encoder", OrdinalEncoder(categories=encoding_cats)),
            ("encoder", EntityEmbedder(embedding_model=embedding_model)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)

    # bin all variables into k bins
    binning_pipeline = Pipeline(
        steps=[
            (
                "binner",
                KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="uniform"),
            )
        ]
    )

    X_train_binned = binning_pipeline.fit_transform(X_train, y_train)
    X_test_binned = binning_pipeline.transform(X_test)

    n_inst_train = X_train_binned.shape[0]
    n_inst_test = X_test_binned.shape[0]
    n_var = X_train.shape[1]

    instances_train = []
    for i in range(0, n_inst_train):
        row = X_train_binned.getrow(i)
        row_reshaped = row.reshape(n_bins, n_var, order="F")
        instances_train.append(row_reshaped.todense())

    instances_test = []
    for i in range(0, n_inst_test):
        row = X_test_binned.getrow(i)
        row_reshaped = row.reshape(n_bins, n_var, order="F")
        instances_test.append(row_reshaped.todense())

    # reshape train and test sets
    X_train_final = np.array(instances_train).reshape(n_inst_train, n_bins, n_var, 1)
    X_test_final = np.array(instances_test).reshape(n_inst_test, n_bins, n_var, 1)

    # build CNN
    model = Sequential()  # add model layers
    model.add(
        Conv2D(64, kernel_size=2, activation="relu", input_shape=(n_bins, n_var, 1))
    )
    model.add(Conv2D(32, kernel_size=2, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train_final, y_train, validation_data=(X_test_final, y_test), epochs=3)

    preds = model.predict_classes(X_test_final)
    print(classification_report(y_test, preds))


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

# test different orderings of columns (random sorting)
