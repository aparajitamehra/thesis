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

from sklearn.metrics import roc_curve

from kerastuner import Objective
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from kerasformain import plot_cm, evaluate_metrics

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


from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from kerastuner import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

from kerasformain import prephybrid
import time
LOG_DIR = f"{int(time.time())}"

n_num = 0
n_cat = 0
binsize = 10


def buildtunedhybrid(hp):
    # numerical
    num_input = tf.keras.Input(shape=(n_num, binsize, 1))
    # x = num_input
    filters = hp.Choice('filters', values=[4, 8, 16, 32])
    # kernel_size = hp.Choice('kernel_size', [(1,1),(3,3),(5,5)])

    conv11 = tf.keras.layers.Convolution2D(filters, kernel_size=(2, 2), activation='relu')(num_input)

    if hp.Choice('pooling_', values=['avg', 'max']) == 'max':
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

    dense21 = tf.keras.layers.Dense(6, activation='relu')(cat_input)

    # merging
    merge = tf.keras.layers.concatenate([flat1, dense21])
    hidden1 = tf.keras.layers.Dense(10, activation='relu')(merge)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden1)

    model = tf.keras.Model(inputs=[num_input, cat_input], outputs=output)

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(
                      hp.Choice('learning_rate',
                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    '''
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    '''
    return model



def main(data_path, descriptor_path,  embedding_model, ds_name):

    global n_num
    global n_cat
    global binsize
    clf = "Multi-Input"

    X, y, X_train, X_test, y_train, y_test = load_credit_scoring_data(
        data_path, descriptor_path, rearrange=True)
    oversampler = RandomOverSampler(sampling_strategy=0.8)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

    X_train_num, y_train, X_test_num, y_test, X_val_num, y_val, X_train_cat, X_test_cat, X_val_cat, n_num, n_cat = prephybrid(
        X_train, X_test, y_train, y_test, binsize=10)


    tuner = RandomSearch(
        buildtunedhybrid,
        objective=Objective("val_auc", direction="max"),
        max_trials=100,
        executions_per_trial =2,
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
    tuner.search([X_train_num, X_train_cat], y_train,
                 validation_data= ([X_val_num,X_val_cat], y_val),
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

    proba_preds_test = best_model.predict([X_test_num, X_test_cat])

    proba_preds_test = proba_preds_test[:, 0]

    class_preds_test = proba_preds_test.round()

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