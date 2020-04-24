import keras
from imblearn.over_sampling import RandomOverSampler

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD

from keras.models import load_model
from pandas import np
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from utils.data_loading import load_credit_scoring_data
from old_scripts.draft_scripts.preprocessing import HighVIFDropper
from utils.entity_embedding import EntityEmbedder


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

early_stopping_auc = keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=20, mode="max", restore_best_weights=True
)

early_stopping_fn = keras.callbacks.EarlyStopping(
    monitor="val_fn", patience=20, mode="min", restore_best_weights=True
)


def main_mlp_keras(data_path, descriptor_path, embedding_model, ds_name):
    # set up data
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path
    )

    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2, random_state=42,
    )

    # set up preprocessing pipeline
    categorical_features = X.select_dtypes(include=("category", "bool")).columns
    numeric_features = X.select_dtypes("number").columns
    encoding_cats = [sorted(X[i].unique().tolist()) for i in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer_num", SimpleImputer()),
            ("highVifDropper", HighVIFDropper(threshold=10)),
            ("scaler", RobustScaler()),
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

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    xdim = X_train.shape[-1]

    # set up logistic regression as a preprocessing + classifier pipeline
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    output_bias = keras.initializers.RandomUniform(initial_bias)

    # build MLP
    model = Sequential()
    model.add(
        Dense(
            64,
            activation="relu",
            input_shape=(xdim,),
            kernel_initializer="uniform",
            bias_initializer=output_bias,
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    adam = Adam(learning_rate=0.01)
    sgd = SGD(learning_rate=0.1, momentum=0.8)

    model.compile(
        loss="binary_crossentropy", optimizer=adam, metrics=METRICS,
    )

    # fit pipeline to training data
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        callbacks=[early_stopping_auc, early_stopping_fn],
    )

    # get score and classification report for test data
    preds = model.predict_classes(X_test)

    model.save(f"models/mlp_keras_{ds_name}.h5")

    # get best score and parameters and classification report for training data
    with open(f"mlp_keras_results_{ds_name}.txt", "w") as f:
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, model.predict_classes(X_train))}\n"
        )
        f.write(f"Scores for test set: " f"{classification_report(y_test, preds)}\n")
        f.write(f"model metric values: {model.evaluate(X_test, y_test)}\n")
        f.write(f"model metric names: {model.metrics_names}\n")


if __name__ == "__main__":
    from pathlib import Path

    for ds_name in ["UK"]:
        print(ds_name)
        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}_new.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main_mlp_keras(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )
