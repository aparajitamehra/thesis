from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, recall_score, roc_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    auc,
)

from utils.data import load_credit_scoring_data
from utils.preprocessing import preprocessing_pipeline_onehot

# MLP using scikitlearn modules

# load data
data = load_credit_scoring_data(
    "datasets/UK/input_UK.csv", "datasets/UK/descriptor_UK.csv"
)

# create train and set sets
y = data.pop("censor")
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# preprocess X_train using pipeline
onehot_preprocessor = preprocessing_pipeline_onehot(X_train)
X_train = onehot_preprocessor.fit_transform(X_train)

# create mlp classifier
mlp = MLPClassifier(
    max_iter=7000,
    activation="relu",
    alpha=0.01,
    hidden_layer_sizes=(5, 2),
    learning_rate="constant",
    momentum=0.8,
    solver="lbfgs",
    early_stopping=True,
)


# set up customer auc function
def custom_auc(x, y):
    fpr, tpr, _ = roc_curve(x, y, pos_label=1)
    return auc(fpr, tpr)


# define my_auc using custom function
my_auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)

# define grid search parameter space
parameter_space = {
    "hidden_layer_sizes": [
        (5, 2),
        (20, 2),
        (50, 2),
        (75, 2),
        (100, 2),
        (50, 20, 2),
        (20, 20, 2),
        (10, 10, 2),
    ],
    "activation": ["relu", "tanh"],
    "solver": ["sgd", "lbfgs", "adam"],
    "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
    "momentum": [0.6, 0.8, 1.0],
    "learning_rate": ["constant", "adaptive"],
}
# define scorers for fitting classifier
scorers = {
    "recall_score": make_scorer(recall_score),
    "f1_score": make_scorer(f1_score),
    "accuracy_score": make_scorer(accuracy_score),
    "my_auc": my_auc,
}

# set up grid search classifier with given parameter space and scorers
clf = GridSearchCV(
    mlp, param_grid=parameter_space, n_jobs=-1, cv=2, scoring=scorers, refit="my_auc"
)

# fit grid search classifier using X_train and y_train, output best parameters
clf.fit(X_train, y_train)
print("Best parameters found:\n", clf.best_params_)
print("My AUC Score:\n", clf.best_score_)

# create confusion matrix with predictions on y_test
X_test = onehot_preprocessor.fit_transform(X_test)
y_pred = clf.predict(X_test)

labels = [0, 1]
print(confusion_matrix(y_test, y_pred, labels=labels))
print(classification_report(y_test, y_pred))
