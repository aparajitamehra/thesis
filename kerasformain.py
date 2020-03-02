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
from utils.preprocessing import preprocessing_pipeline_onehot, MLPpreprocessing_pipeline_onehot #CNNpreprocessing_pipeline_onehot
from sklearn.model_selection import train_test_split, GridSearchCV

from keras.preprocessing import sequence
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import tempfile
from scipy.sparse import csr_matrix, isspmatrix

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    auc,
)



def prepmlp(X_train, X_test, y_train, y_test):




    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

# mlp preprocessing
    onehot_preprocessor = MLPpreprocessing_pipeline_onehot(X_train)
    onehot_preprocessor.fit(X_train)

    X_train = onehot_preprocessor.transform(X_train)
    X_test = onehot_preprocessor.transform(X_test)
    X_val = onehot_preprocessor.transform(X_val)

    if isspmatrix(X_train):
        X_train = X_train.todense()
    if isspmatrix(X_test):
        X_test = X_test.todense()
    if isspmatrix(X_val):
        X_val = X_val.todense()

# min max capping
    X_train = np.clip(X_train, -5, 5)
    X_val = np.clip(X_val, -5, 5)
    X_test = np.clip(X_test, -5, 5)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return X_train,y_train,X_test,y_test, X_val, y_val


def prepcnn(X_train, X_test, y_train, y_test):


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

    onehot_preprocessor = CNNpreprocessing_pipeline_onehot(X_train)
    onehot_preprocessor.fit(X_train)
    X_train = onehot_preprocessor.transform(X_train)
    X_test = onehot_preprocessor.transform(X_test)
    X_val = onehot_preprocessor.transform(X_val)


    print("type",type(X_train))

    if isspmatrix(X_train):
        X_train = X_train.toarray()
    if isspmatrix(X_test):
        X_test = X_test.toarray()
    if isspmatrix(X_val):
        X_val = X_val.toarray()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    y_val = y_val.to_numpy()

    print("pre func: ", X_train.shape)


    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))

    y_train = y_train.reshape(y_train.shape[0], )
    y_test = y_test.reshape(y_test.shape[0], )
    y_val = y_val.reshape(y_val.shape[0], )

    print("post func: ", X_train.shape)
    return X_train,y_train,X_test,y_test, X_val, y_val


def make_cnn(metrics, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    print('Build model...')
    model = keras.Sequential([
        keras.layers.Conv1D(filters=8, kernel_size=2, activation='relu', input_shape=(None,1), padding='same', strides=1),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),

    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-2),
              metrics=metrics)
    return model

def make_MLP(xdim, metrics, output_bias=None):


    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    print('Build model...')
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(xdim,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),

    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-2),
                  metrics=metrics)

    return model

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

def plot_metrics(history, modtype, colors):

    mpl.rcParams['figure.figsize'] = (12, 10)
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors, label=modtype + ' Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors, linestyle="--", label=modtype+' Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.5, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

def makeweightedMLP(X_train, X_test, y_train, y_test):

    print("makemlp")
    X_train, y_train, X_test, y_test, X_val, y_val = prepmlp(X_train, X_test, y_train, y_test)
    xdim = X_train.shape[-1]
    EPOCHS = 100
    BATCH_SIZE = 2000

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # initial weights

    neg, pos = np.bincount(y_train)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    initial_bias = np.log([pos / neg])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    model = make_MLP(xdim=X_train.shape[-1],metrics=METRICS)
    model.summary()

    '''
    baseline_history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val))
    '''
    # set initial bias
    model = make_MLP(xdim, METRICS,output_bias=initial_bias)
    model.predict(X_train[:10])
    results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)
    '''
    # zero bias versus bias init test
    model = make_MLP(xdim, METRICS)
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0])
    zero_bias_history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=0)

    model = make_MLP(xdim,METRICS)
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0])
    careful_bias_history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=0)

    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



    train_predictions_baseline = model.predict(X_train, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(X_test, batch_size=BATCH_SIZE)

    baseline_results = model.evaluate(X_test, y_test,
                                      batch_size=BATCH_SIZE, verbose=0)

    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    '''
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    weighted_model = make_MLP(xdim, METRICS)
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),

        class_weight=class_weight)


    train_predictions_weighted = weighted_model.predict(X_train, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(X_test, batch_size=BATCH_SIZE)

    weighted_results = weighted_model.evaluate(X_test, y_test,
                                               batch_size=BATCH_SIZE, verbose=0)

    print("Weighted model: ")
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ': ', value)
    print()

    return weighted_history

def make_weightedCNN(X_train, X_test, y_train, y_test):
    print("makecnn")
    print("pre: ", X_train.shape)
    X_train, y_train, X_test, y_test, X_val, y_val = prepcnn(X_train, X_test, y_train, y_test)
    print("post: ", X_train.shape)
    EPOCHS = 100
    BATCH_SIZE = 2000

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision()
    ]

    # initial weights

    neg, pos = np.bincount(y_train)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    initial_bias = np.log([pos / neg])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    model = make_cnn(metrics=METRICS)
    model.summary()

    '''
    baseline_history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val))
    '''
    # set initial bias
    model = make_cnn(METRICS, output_bias=initial_bias)
    model.predict(X_train[:10])
    results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)
    '''
    # zero bias versus bias init test
    model = make_MLP(xdim, METRICS)
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0])
    zero_bias_history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=0)

    model = make_MLP(xdim,METRICS)
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0])
    careful_bias_history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=0)

    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



    train_predictions_baseline = model.predict(X_train, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(X_test, batch_size=BATCH_SIZE)

    baseline_results = model.evaluate(X_test, y_test,
                                      batch_size=BATCH_SIZE, verbose=0)

    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    '''
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    weighted_model = make_cnn(METRICS)
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),

        class_weight=class_weight)

    train_predictions_weighted = weighted_model.predict(X_train, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(X_test, batch_size=BATCH_SIZE)

    weighted_results = weighted_model.evaluate(X_test, y_test,
                                               batch_size=BATCH_SIZE, verbose=0)


    print("Weighted model: ")
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ': ', value)
    print()



    return weighted_history







