import csv

import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

from sklearn.model_selection import train_test_split
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
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

np.random.seed = 50

early_stopping_auc = keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=100, mode="max", restore_best_weights=True
)


# main cnn function
def main_cnn_trans(data_path, descriptor_path, embedding_model, ds_name):

    # load and split data
    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path, rearrange="Emb"
    )

    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2, random_state=42,
    )

    n_bins = 10

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
    X_val = preprocessor.transform(X_val)

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
    X_val_binned = binning_pipeline.transform(X_val)

    # define shape variables
    n_inst_train = X_train_binned.shape[0]
    n_inst_test = X_test_binned.shape[0]
    n_inst_val = X_val_binned.shape[0]
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

    # add a n_bin x n_var matrix for each validation instance to list
    instances_val = []
    for i in range(0, n_inst_val):
        row = X_train_binned.getrow(i)
        row_reshaped = row.reshape(n_bins, n_var, order="F")
        row_dense = row_reshaped.todense()
        instances_val.append(row_dense)

    # reformat instances from lists into arrays, so they can be shuffled in unison
    instances_train = np.array(instances_train)
    instances_val = np.array(instances_val)
    instances_test = np.array(instances_test)

    # # shuffle columns of instance matrices to change spatial relationships
    # all_instances = list(zip(instances_train.T, instances_val.T, instances_test.T))
    # np.random.shuffle(all_instances)
    # instances_train, instances_val, instances_test = zip(*all_instances)
    #
    # # reformat instances back to arrays after shuffle
    # instances_train = np.array(instances_train).T
    # instances_val = np.array(instances_val).T
    # instances_test = np.array(instances_test).T

    # reshape train, test and validation sets to make them appropriate input to CNN
    X_train_final = instances_train.reshape(n_inst_train, n_bins, n_var, 1)
    X_test_final = instances_test.reshape(n_inst_test, n_bins, n_var, 1)
    X_val_final = instances_val.reshape(n_inst_val, n_bins, n_var, 1)

    neg, pos = np.bincount(y_train)
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )
    initial_bias = np.log([pos / neg])

    output_bias = keras.initializers.RandomNormal(initial_bias)

    # build CNN
    model = Sequential()  # add model layers
    model.add(
        Conv2D(
            filters=6,
            kernel_size=(4, 8),
            padding="same",
            activation="relu",
            input_shape=(n_bins, n_var, 1),
            kernel_initializer="normal",
            bias_initializer=output_bias,
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
        validation_data=(X_val_final, y_val),
        epochs=1000,
        batch_size=64,
        callbacks=[early_stopping_auc],
    )

    class_preds = model.predict_classes(X_test_final)
    proba_preds = model.predict_proba(X_test_final)

    model.save(f"models/cnn_emb_{ds_name}.h5")
    model.save_weights(f"models/weights/cnn_emb_weights_{ds_name}.h5")

    with open(f"results_plots/cnn_results/cnn_emb_results_{ds_name}.txt", "w") as f:
        f.write(
            f"Scores for train set: \n"
            f"{classification_report(y_train, model.predict_classes(X_train_final))}\n"
        )
        f.write(
            f"Scores for test set: " f"{classification_report(y_test, class_preds)}\n"
        )
        f.write(f"model metric values: {model.evaluate(X_test_final, y_test)}\n")
        f.write(f"model metric names: {model.metrics_names}\n")

    # get evaluation metrics for test data

    with open(
        f"../results_plots/cnn_results/cnn_emb_metrics.csv", "a", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([ds_name, "cnn_emb", "AUC", roc_auc_score(y_test, proba_preds)])
        writer.writerow([ds_name, "cnn_emb", "F1_score", f1_score(y_test, class_preds)])
        writer.writerow(
            [ds_name, "cnn_emb", "F_beta", fbeta_score(y_test, class_preds, beta=3)]
        )
        writer.writerow(
            [ds_name, "cnn_emb", "Accuracy", accuracy_score(y_test, class_preds)]
        )
        writer.writerow(
            [
                ds_name,
                "cnn_emb",
                "Balanced_Accuracy",
                balanced_accuracy_score(y_test, class_preds),
            ]
        )
        writer.writerow(
            [ds_name, "cnn_emb", "Precision", precision_score(y_test, class_preds)]
        )
        writer.writerow(
            [ds_name, "cnn_emb", "Recall", recall_score(y_test, class_preds)]
        )

    fig = plt.figure()
    cm = confusion_matrix(y_test, class_preds > 0.5)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"confusion matrix cnn_emb_{ds_name}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    fig.savefig(f"results_plots/cnn_results/cnn_emb_{ds_name}_CM.png")
    plt.close(fig)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, proba_preds)
    roc_auc = roc_auc_score(y_test, proba_preds)

    # plot ROC curve
    fig = plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC CNN_Emb_{ds_name}")
    plt.legend(loc="lower right")
    fig.savefig(f"results_plots/cnn_results/cnn_emb_{ds_name}_ROC.png")
    plt.close(fig)


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
