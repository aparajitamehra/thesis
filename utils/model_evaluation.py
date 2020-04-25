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
    auc,
)
from scikitplot.metrics import plot_ks_statistic
from scikitplot.helpers import binary_ks_curve

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
    "test_ks_stat",
]


def evaluate_sklearn(scores, clf_name, model, ds_name):

    with open(f"results/{clf_name}/{clf_name}_results_{ds_name}.txt", "w") as f:
        f.write(f"Non-Nested auc_score: {model.best_score_:.3f}\n")
        # f.write(
        #     f"AUC score on hold-out test set: {roc_auc_score(y_test, proba_preds)} "
        # )
        f.write(f"Best parameter set: {model.best_params_}\n")
        f.write(f"Best scores index: {model.best_index_}\n")

    pd.DataFrame(scores)[key_results].to_csv(
        f"results/{clf_name}/{clf_name}_CV_results_{ds_name}.csv"
    )


def evaluate_keras(y_test, class_preds, proba_preds, clf_name, ds_name, iter):

    # calculate ks_stat
    res = binary_ks_curve(y_test, proba_preds)
    ks_stat = res[3]

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
            [
                ds_name,
                clf_name,
                iter,
                "Brier_score",
                brier_score_loss(y_test, class_preds),
            ]
        )
        writer.writerow([ds_name, clf_name, iter, "KS_stat", ks_stat])


def plot_cm(labels, class_preds, clf_name, modelname, iter, p=0.5):
    fig = plt.figure()
    cm = confusion_matrix(labels, class_preds > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"confusion matrix {modelname}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    fig.savefig(f"results/{clf_name}/{modelname}_CM{iter}.png")
    plt.close(fig)


def plot_roc_sklearn(labels, proba_preds, clf_name, modelname):
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
    fig.savefig(f"results/{clf_name}/{modelname}_ROC{iter}.png")
    plt.close(fig)


def roc_iter(y_test, proba_preds, tprs, mean_fpr, aucs, iter):
    fpr, tpr, thresholds = roc_curve(y_test, proba_preds)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(
        fpr, tpr, lw=1, alpha=0.3, label="ROC fold %d (AUC = %0.2f)" % (iter, roc_auc)
    )


def plot_roc(tprs, aucs, mean_fpr, clf_name, modelname):
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title(f"Cross-Validation ROC {modelname}", fontsize=18)
    plt.legend(loc="lower right", prop={"size": 15})


def plot_KS(y_test, proba_preds, clf_name, modelname, iter):
    fig, ax = plt.subplots()
    plot_ks_statistic(y_test, proba_preds, ax=ax)
    fig.savefig(f"results/{clf_name}/{modelname}_KS{iter}.png")
    plt.close(fig)
