import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
)

# define key results to show in cv_results
key_results = [
    "test_auc",
    "test_f1_score",
    "test_fbeta_score",
    "test_accuracy_score",
    "test_balanced_accuracy",
    "test_precision_score",
    "test_recall_score",
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
            [ds_name, clf_name, iter, "F1 score", f1_score(y_test, class_preds)]
        )
        writer.writerow(
            [
                ds_name,
                clf_name,
                iter,
                "F beta",
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
                "Balanced Accuracy",
                balanced_accuracy_score(y_test, class_preds),
            ]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "Precision", precision_score(y_test, class_preds)]
        )
        writer.writerow(
            [ds_name, clf_name, iter, "Recall", recall_score(y_test, class_preds)]
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
