from __future__ import print_function


from utils.data import load_credit_scoring_data

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import numpy as np
from numpy import newaxis
from utils.preprocessing import preprocessing_pipeline_onehot, MLPpreprocessing_pipeline_onehot,cnn2dprep_num, EntityPrep #CNNpreprocessing_pipeline_onehot
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from kerasformain import plot_cm, evaluate_metrics, prep2dcnn

import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import tempfile
from scipy.sparse import csr_matrix, isspmatrix

from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, fbeta_score, balanced_accuracy_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    auc,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import model_from_json

from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from kerastuner import HyperModel, Objective
from kerastuner.engine.hyperparameters import HyperParameters


from kerasformain import prephybrid
import time

LOG_DIR = f"{int(time.time())}"


nfeats =0
binsize = 10


def buildtuned2DCNN(hp):
    filters = hp.Choice('filters', values=[4, 8, 16, 32])
    #kernel = hp.Choice('kernel_size', values = [2,3])

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filters=filters, kernel_size=2, activation='relu', input_shape=(nfeats, binsize, 1),
                            padding='same'))

    if hp.Choice('pooling_', values=['avg', 'max']) == 'max':
        model.add(keras.layers.MaxPooling2D(pool_size=2))
    else:
        model.add(keras.layers.AveragePooling2D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))


    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4, 1e-5])),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def main(data_path, descriptor_path, embedding_model, ds_name):
    global nfeats
    global binsize
    clf="2DCNN"

    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True)

    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)


    print("binsize: {}".format(binsize))
    X_train, y_train, X_test, y_test, X_val, y_val, nfeats = prep2dcnn(X_train, X_test, y_train, y_test, binsize=binsize)

    tuner = RandomSearch(
        buildtuned2DCNN,
        objective= Objective("val_auc", direction="max"),
        max_trials=100,
        executions_per_trial=2,
        directory='results_plots/{}'.format(clf),
        project_name='{}_tuning'.format(ds_name)
    )


    '''

    tuner = Hyperband(
        secbuildtunedhybrid,
        max_epochs=100,
        objective='val_accuracy',
        seed=1,
        executions_per_trial=2,
        directory='hyperband',
        project_name='thesis',

    )

    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
    )
    '''
    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=100,
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

    for ds_name in ["UK","bene1","bene2","german"]:
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