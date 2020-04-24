import sys

from sklearn.metrics import make_scorer, confusion_matrix
import pandas as pd

from sklearn.metrics import roc_curve


def total_cost_scorer(y_true, y_pred, cost_FP, cost_FN):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    cost_FP = cost_FP
    cost_FN = cost_FN

    total_cost = 0 * (tn + tp) + cost_FP * fp + cost_FN * fn

    return total_cost


make_scorer(total_cost_scorer, totals=totals_data, greater_is_better=False)

# def EMProcInfo(scores, classes):
#     if len(scores) != len(classes):
#         sys.exit('Length of scores and classes vectors is not equal')
#
#     prediction = prediction(scores, classes)
#     perf = performance(prediction, "rch")
#     n0 < - prediction @ n.pos[[1]]
#     n1 < - prediction @ n.neg[[1]]
#     pi0 < - n0 / (n0 + n1)
#     pi1 < - n1 / (n0 + n1)
#     F0 < - perf @ y.values[[1]]
#     F1 < - perf @ x.values[[1]]
#     return n0=n0, n1=n1, pi0=pi0, pi1=pi1, F0=F0, F1=F1
#
# def EMP(scores, classes, p0=0.55, p1=0.1, ROI=0.2644):
#     roc =EMProcInfo(scores, classes)
#
#     alpha < - 1 - p0 - p1
#
#     lambda = (0, (roc$pi1 * ROI / roc$pi0)*diff(roc$F1) / diff(roc$F0)
#     lambda < - c( lambda[lambda < 1], 1)
#
#     lambdaii < - head( lambda, n=-1)
#     lambdaie < - tail(lambda , n=-1)
#     F0 < - roc$F0[1:length(lambdaii)]
#     F1 < - roc$F1[1:length(lambdaii)]
#
#     EMPC < - sum(alpha * (lambdaie - lambdaii) * (roc$pi0 * F0 * (lambdaie+lambdaii) / 2 - ROI * F1 * roc$pi1)) +
#     (roc$pi0 * tail(F0, n=1) - ROI * roc$pi1 * tail(F1, n=1))*p1
#
#
# EMPCfrac < - sum(alpha * (lambdaie - lambdaii) * (roc$pi0 * F0+roc$pi1 * F1)) +
# p1 * (roc$pi0 * tail(F0, n=1) + roc$pi1 * tail(F1, n=1))
#
# list(EMPC=EMPC, EMPCfrac=EMPCfrac)
#
#
# make_scorer(EMP, greater_is_better=True)
