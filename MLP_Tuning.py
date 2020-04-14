
from __future__ import print_function

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D
from utils.data import load_credit_scoring_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import numpy as np
from numpy import newaxis
from utils.preprocessing import preprocessing_pipeline_onehot, MLPpreprocessing_pipeline_onehot,cnn2dprep_num, EntityPrep #CNNpreprocessing_pipeline_onehot
from sklearn.model_selection import train_test_split, GridSearchCV

from keras.preprocessing import sequence
from sklearn.metrics import roc_curve
import tensorflow as tf
from kerastuner import Objective
from tensorflow import keras


from keras.utils import plot_model
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import tempfile
from scipy.sparse import csr_matrix, isspmatrix
import talos
from talos.utils import lr_normalizer
from tensorflow.keras.utils import plot_model
from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, fbeta_score, balanced_accuracy_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    auc,
)

from keras.models import model_from_json

from kerastuner.tuners import RandomSearch

from kerasformain import prepmlp, plot_cm, evaluate_metrics, prep_ent_mlp
import time
LOG_DIR = f"{int(time.time())}"

xdim=0

def buildtunedMLP(hp):

    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(xdim, ) ))

    model.add(keras.layers.Dense(units=hp.Int('units',
                                              min_value=2,
                                              max_value=16,
                                              step=2),
                                 activation='relu'))


    for i in range(hp.Int('nr_layers', 1, 3)):
        model.add(keras.layers.Dense(units=hp.Int('units',
                                    min_value=2,
                                    max_value=16,
                                    step=2),
                                    activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

    model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Choice('learning_rate',
                                    values=[1e-2, 1e-3, 1e-4])),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]

    )

    return model


def main(data_path, descriptor_path,  embedding_model, ds_name):

    clf = "keras_MLP"

    global xdim


    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path)
    oversampler = RandomOverSampler(sampling_strategy=0.8)

    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

    X_train, y_train, X_test, y_test, X_val, y_val = prepmlp(X_train, X_test, y_train, y_test)
    print("X_trainshape: {}".format(X_train.shape))
    xdim = X_train.shape[-1]
    print("X_dim: {}".format(xdim))


    tuner = RandomSearch(
        buildtunedMLP,
        objective= Objective("val_auc", direction="max"),
        max_trials=1,
        executions_per_trial =1,
        directory='results_plots/{}'.format(clf),
        project_name='thesis_tuning'
    )


    tuner.search(X_train, y_train,
                 validation_data= (X_val, y_val),
                 epochs=10,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                            patience=10)],
                 )


    best_model = tuner.get_best_models(1)[0]

    model_json = best_model.to_json()

    with open("results_plots/{}/{}_model.json".format(clf, ds_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    best_model.save_weights('results_plots/{}/{}_model.h5'.format(clf, ds_name))

    proba_preds_test = best_model.predict(X_test)
    class_preds_test = best_model.predict_classes(X_test)

    plot_cm(y_test, class_preds_test, modelname=clf, dsname=ds_name)
    plot_model(model=best_model, to_file='results_plots/{}/{}_model_plot.png'.format(clf, ds_name), show_shapes=True)

    evaluate_metrics(proba_preds_test, class_preds_test, y_test, clf_name=clf, ds_name=ds_name)



if __name__ == "__main__":
    from pathlib import Path
    for ds_name in ["UK"]:
        print(ds_name)

        embedding_model = None
        embedding_model_path = f"datasets/{ds_name}/embedding_model_{ds_name}.h5"
        if Path(embedding_model_path).is_file():
            embedding_model = load_model(embedding_model_path)

        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
            embedding_model,
            ds_name,
        )

    plt.show()