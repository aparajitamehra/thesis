import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from scipy.stats import ks_2samp

# define key results to show in cv_results
key_results = [
    "test_auc",
    "test_f1_score",
    "test_fbeta_score",
    "test_accuracy_score",
    "test_balanced_accuracy",
    "test_precision_score",
    "test_recall_score",
    "test_brier_score_loss",
]


def evaluate_sklearn(y_test, proba_preds, scores, clf_name, model, ds_name):
    with open(f"results/{clf_name}/{clf_name}_results_{ds_name}.txt", "w") as f:
        f.write(f"Non-Nested auc_score: {model.best_score_:.3f}\n")
        f.write(
            f"AUC score on hold-out test set: {roc_auc_score(y_test, proba_preds)} "
        )
        f.write(f"Best parameter set: {model.best_params_}\n")
        f.write(f"Best scores index: {model.best_index_}\n")

    pd.DataFrame(scores)[key_results].to_csv(
        f"results/{clf_name}/{clf_name}_CV_results_{ds_name}.csv"
    )


def evaluate_keras(y_test, class_preds, proba_preds, clf_name, ds_name, iter):

    with open(f"results/{clf_name}/{clf_name}_metrics.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [ds_name, clf_name, iter, "AUC", roc_auc_score(y_test, proba_preds)]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "F1_score", f1_score(y_test, class_preds)]
        )
        writer.writerow(
            [
                ds_name,
                clf_name,
                iter,
                "F_beta",
                fbeta_score(y_test, class_preds, beta=3),
            ]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "Accuracy", accuracy_score(y_test, class_preds)]
        )
        writer.writerow(
            [
                ds_name,
                clf_name,
                iter,
                "Balanced_Accuracy",
                balanced_accuracy_score(y_test, class_preds),
            ]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "Precision", precision_score(y_test, class_preds)]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "Recall", recall_score(y_test, class_preds)]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "Brier_score", brier_score_loss(y_test, class_preds)]
        )


def plot_cm(labels, class_preds, clf_name, modelname, iter, p=0.5):
    fig = plt.figure()
    cm = confusion_matrix(labels, class_preds > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"confusion matrix {modelname}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    fig.savefig(f"results/{clf_name}/{modelname}_CM_{iter}.png")
    plt.close(fig)


def plot_roc(labels, proba_preds, clf_name, modelname, iter):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels, proba_preds)
    roc_auc = roc_auc_score(labels, proba_preds)

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
    plt.title(f"ROC {modelname}")
    plt.legend(loc="lower right")
    fig.savefig(f"results/{clf_name}/{modelname}_ROC_{iter}.png")
    plt.close(fig)


def make_ks_plot(y_train, train_proba, y_test, test_proba, clf_name, modelname, iter):
    """NOT WORKING RIGHT NOW
    This is from an example I found online:
    https://www.cerebriai.com/testing-for-overfitting-in-binary-classifiers/
    It is the only one I could find to apply KS test to binary classification
    but this code seems to plot the 1/0 class predictions, and apparently KS_test only accepts
    continuous distributions. So the plots created here make it seem like our
    classifiers are really bad. So something is wrong.
    """
    bins = 30
    fig_sz = (10, 8)

    train = pd.DataFrame(y_train)
    test = pd.DataFrame(y_test)
    train["probability"] = train_proba
    test["probability"] = test_proba
    train[["censor", "probability"]] *= 1
    print(train.head())
    print(test.head())

    decisions = []
    for df in [train, test]:
        d1 = df['probability'][df["censor"] == 1]
        d2 = df['probability'][df["censor"] == 0]
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    fig = plt.figure(figsize=fig_sz)

    train_pos = plt.hist(decisions[0],
                         color='r', alpha=0.5, range=low_high, bins=bins,
                         histtype='stepfilled', density=True,
                         label='+ (train)')

    train_neg = plt.hist(decisions[1],
                         color='b', alpha=0.5, range=low_high, bins=bins,
                         histtype='stepfilled', density=True,
                         label='- (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    test_pos = plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='+ (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    test_neg = plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='- (test)')

    # get the KS score
    ks = ks_2samp(decisions[0], decisions[2])
    print(ks)

    plt.xlabel("Classifier Output", fontsize=12)
    plt.ylabel("Arbitrary Normalized Units", fontsize=12)

    plt.xlim(0, 1)
    plt.plot([], [], ' ', label='KS Statistic (p-value) :' + str(round(ks[0], 2)) + '(' + str(round(ks[1], 2)) + ')')
    plt.legend(loc='best', fontsize=12)
    fig.savefig(f"results/{clf_name}/{modelname}_KS_{iter}.png")
    plt.close(fig)
