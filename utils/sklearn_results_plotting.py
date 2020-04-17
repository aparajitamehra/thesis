import numpy as np
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
    classification_report,
)

# define key results to show in cv_results
key_results = [
    "mean_test_recall_score",
    "mean_test_f1_score",
    "mean_test_accuracy_score",
    "mean_test_precision_score",
    "mean_test_auc",
    "mean_test_fbeta_score",
    "mean_test_balanced_accuracy",
]


def best_model_parameters(
    X_train, y_train, y_test, proba_preds, nested_score, clf_name, model, ds_name
):
    with open(
        f"results_plots/sklearn_results/{clf_name}_results_{ds_name}.txt", "w"
    ) as f:
        f.write(f"Best auc_score on train set: {model.best_score_:.3f}\n")
        f.write(f"Best auc_score on test set: {roc_auc_score(y_test, proba_preds)}\n")
        f.write(f"Best parameter set: {model.best_params_}\n")
        f.write(f"Best scores index: {model.best_index_}\n")
        f.write(
            f"Scores for train set: "
            f"{classification_report(y_train, model.predict(X_train))}\n"
        )
        f.write(f"Nested Scores: {nested_score.mean()}\n")

    pd.DataFrame(model.cv_results_)[key_results].to_csv(
        f"results_plots/sklearn_results/{clf_name}_CV_results_{ds_name}.csv"
    )


def evaluate_metrics(y_test, class_preds, proba_preds, clf_name, ds_name):
    with open(
        f"results_plots/sklearn_results/{clf_name}_metrics.csv", "a", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([ds_name, clf_name, "AUC", roc_auc_score(y_test, proba_preds)])
        writer.writerow([ds_name, clf_name, "F1_score", f1_score(y_test, class_preds)])
        writer.writerow(
            [ds_name, clf_name, "F_beta", fbeta_score(y_test, class_preds, beta=3)]
        )
        writer.writerow(
            [ds_name, clf_name, "Accuracy", accuracy_score(y_test, class_preds)]
        )
        writer.writerow(
            [
                ds_name,
                clf_name,
                "Balanced_Accuracy",
                balanced_accuracy_score(y_test, class_preds),
            ]
        )
        writer.writerow(
            [ds_name, clf_name, "Precision", precision_score(y_test, class_preds)]
        )
        writer.writerow(
            [ds_name, clf_name, "Recall", recall_score(y_test, class_preds)]
        )


def plot_cm(labels, class_preds, modelname, p=0.5):
    fig = plt.figure()
    cm = confusion_matrix(labels, class_preds > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"confusion matrix {modelname}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    fig.savefig(f"results_plots/sklearn_plots/{modelname}_CM.png")
    plt.close(fig)


def plot_roc(labels, class_preds, proba_preds, modelname):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels, proba_preds[:, 1])
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
    fig.savefig(f"results_plots/sklearn_plots/{modelname}_ROC.png")
    plt.close(fig)


def plot_gridsearch_metrics(
    results, grid_param, scoring,
):
    plt.xlabel(f"{grid_param}")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0.1, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[f"param_{grid_param}"].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ["g", "k", "b", "r", "c"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [X_axis[best_index]] * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
