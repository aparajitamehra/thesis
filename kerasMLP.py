
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
from utils.preprocessing import preprocessing_pipeline_onehot, MLPpreprocessing_pipeline_onehot
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


# load data
data = load_credit_scoring_data(
    "datasets/UK/input_UK.csv", "datasets/UK/descriptor_UK.csv"
)




y = data.pop("censor")
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

#mlp preprocessing
onehot_preprocessor = MLPpreprocessing_pipeline_onehot(X_train)
onehot_preprocessor.fit(X_train)
X_train = onehot_preprocessor.transform(X_train).todense()
X_test = onehot_preprocessor.transform(X_test).todense()
X_val = onehot_preprocessor.transform(X_val).todense()

# min max capping
X_train = np.clip(X_train, -5, 5)
X_val = np.clip(X_val, -5, 5)
X_test= np.clip(X_test, -5, 5)


y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_val = np.asarray(y_val)

xdim=X_train.shape[-1]

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

#initial weights

neg, pos = np.bincount(y_train)
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
initial_bias = np.log([pos/neg])



def make_MLP(xdim, metrics = METRICS, output_bias = None, ):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)


    print('Build model...')
    model = keras.Sequential([
        keras.layers.Dense(16, activation ='relu', input_shape=(xdim,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),

    ])

    model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-2),
              metrics=metrics)
    return model



#baseline model

EPOCHS =100
BATCH_SIZE = 2000

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


model = make_MLP(xdim=X_train.shape[-1])
model.summary()

baseline_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(X_val, y_val))



#set initial bias
model = make_MLP(xdim,output_bias = initial_bias)
model.predict(X_train[:10])
results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)


#zero bias versus bias init test
model = make_MLP(xdim)
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(X_val, y_val),
    verbose=0)


model = make_MLP(xdim)
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

def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

plt.figure(1)
plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)




def plot_metrics(history):

  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.5,1])
    else:
      plt.ylim([0,1])

    plt.legend()

plt.figure(2)
plot_metrics(baseline_history)



train_predictions_baseline = model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(X_test, batch_size=BATCH_SIZE)


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


baseline_results = model.evaluate(X_test, y_test,
                                  batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plt.figure(3, figsize=(5,5))
plot_cm(y_test, test_predictions_baseline)

def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ =roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([50,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

plt.figure(4)
plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


weight_for_0 = (1 / neg)*(total)/2.0
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


weighted_model = make_MLP(xdim)
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(X_val, y_val),
    # The class weights go here
    class_weight=class_weight)

plt.figure(5)
plot_metrics(weighted_history)


train_predictions_weighted = weighted_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(X_test, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(X_test, y_test,
                                           batch_size=BATCH_SIZE, verbose=0)

print("Weighted model: ")
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()



plt.figure(6)
plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right')



plt.show()
