# -------------------------------------- #
# --
# -------------------------------------- #

import numpy as np
from .get_os_conf_mat_terms import get_os_conf_mat_terms

def get_metrics(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'macro'
    , metric = 'precision'):
    """
    Get true negative rate, precision or recall with macro or micro averaging.
    :param y_true: Correct (observed) values
    :param y_pred: Predicted values
    :param unknown_label: label representing unknown class
    :param labels: possible labels for the classes. If None, considers unique values from ``y_true``, ``y_pred``, ``unknown_label``
    :param average: `'micro'` or `'macro'` averaging
    :param metric: either `'precision'`, `'recall'` or `'true_negative_rate'`
    :return:
    """

    assert metric in ['recall', 'precision', 'true_negative_rate']
    # ---------------------- #
    # -- get terms
    # ---------------------- #
    conf_mat_terms = get_os_conf_mat_terms(
        y_true = y_true
        , y_pred = y_pred
        , unknown_label = unknown_label
        , labels = labels
        , return_dict = not True
    )

    tps, fns, fps, tns = conf_mat_terms


    # -- set formula parameters.
    # precision, recall and youdens are on the form of sum(A)/(sum(A) + sum(B)).
    # so I write just one statement.

    if metric == 'precision':
        A = tps
        B = fps
    elif metric == 'recall':
        A = tps
        B = fns
    elif metric == 'true_negative_rate':
        A = tns
        B = fps #[tns[i_]+fps[i_] for i_ in range(len(tns))]

    # -------------------------------------- #
    # get metric average
    # -------------------------------------- #
    # A, B
    if average == 'micro':
        metric_value = sum(A)/(sum(A) + sum(B))
    elif average == 'macro':
        rates = [a/(a+b) for a, b in zip(A, B)]
        # rates = [(a, b) for a, b in zip(A, B)]

        metric_value = np.mean(rates)

    return metric_value


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -- individual metrics -- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---------------------------- #
# -- OS precision
# ---------------------------- #
def os_precision(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'macro'
):

    results = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='precision')

    return results

# ---------------------------- #
# os recall
# ---------------------------- #
def os_recall(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'macro'
):

    results = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='recall')
    return results

# ---------------------------- #
# -- OS TNR
# ---------------------------- #
def os_true_negative_rate(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'macro'
):

    results = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='true_negative_rate')

    return results


# ---------------------------- #
# Youden's index
# ---------------------------- #
def os_youdens_index(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'macro'
):

    tnr = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='true_negative_rate')

    recall = get_metrics(
        y_true=y_true
        , y_pred=y_pred
        , unknown_label=unknown_label
        , labels=labels
        , average=average
        , metric='recall')

    youdens = recall + tnr -1
    return youdens

# ---------------------------- #
# --
# ---------------------------- #

def os_accuracy(
        y_true
        , y_pred
        , unknown_label = -1
        , alpha = 0.5
):
    """
    Normalized accuracy
    Sources:
    Nearest neighbors distance ratio open-set classifier. Pedro R. Mendes Júnior 2015
    https://arxiv.org/pdf/1811.08581.pdf (pag 11/19)

    :param y_true: Observed Values
    :param y_pred: predictions
    :param unknown_label: label representing unknown class
    :param alpha: Regularization parameter. Weight for known Accuracy (unknown Accuracy weight = 1-lambda)
    :return:
    """
    # y_true, y_pred

    # ---------------------------------- #
    # -- AKS accuracy for the knowns -- #
    # ---------------------------------- #
    idx_knowns = np.array(y_true) != unknown_label
    y_true_known = np.array(y_true)[idx_knowns]
    y_pred_known = np.array(y_pred)[idx_knowns]

    y_true_unknown = np.array(y_true)[~idx_knowns]
    y_pred_unknown = np.array(y_pred)[~idx_knowns]


    AKS = sum(y_true_known==y_pred_known) / len(y_true_known)


    # ---------------------------------- #
    # -- accuracy for the unknown -- #
    # ---------------------------------- #
    if len(y_true_unknown) == 0:
        AUS = None
    else:
        AUS = sum(y_true_unknown == y_pred_unknown) / len(y_pred_unknown)

    # ------------------------ #
    # -- normalized accuracy
    # when there are no unknowns, do not consider AUS
    # ------------------------ #
    if len(y_true_unknown) == 0:
        norm_acc = AKS
    else:
        norm_acc = alpha*AKS + (1-alpha)*AUS


    return norm_acc

# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
# from ostools import metrics as osm
# osm.os_precision
# osm.os_recall

