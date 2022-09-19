# -------------------------------------- #
# --
# -------------------------------------- #

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from .get_os_conf_mat_terms import get_os_conf_mat_terms

def get_metrics(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'micro'
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

    metric


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
        B = fps

    # -------------------------------------- #
    # get metric average
    # -------------------------------------- #
    if average == 'micro':
        metric = sum(A)/(sum(A) + sum(B))
    elif average == 'macro':
        rates = [a/(a+b) for a, b in zip(A, B)]
        # rates = [(a, b) for a, b in zip(A, B)]

        metric = np.mean(rates)

    return metric


# ------------------------------------- #
# -- individual metrics
# ------------------------------------- #


def os_precision(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'micro'
):

    results = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='precision')

def os_recall(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'micro'
):

    results = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='recall')


def os_true_negative_rate(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'micro'
):

    results = get_metrics(
            y_true = y_true
            , y_pred = y_pred
            , unknown_label=unknown_label
            , labels=labels
            , average=average
            , metric='os_true_negative_rate')



def os_youdens_index(
    y_true
    , y_pred
    , unknown_label = -1
    , labels=None
    , average = 'micro'
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


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
# from ostools import metrics as osm
# osm.os_precision
# osm.os_recall

