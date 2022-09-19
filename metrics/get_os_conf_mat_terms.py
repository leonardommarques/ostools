# -------------------------------------- #
# --
# -------------------------------------- #

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

def get_os_conf_mat_terms(
        y_true
        , y_pred
        , unknown_label
        , labels=None
        , return_dict = False
):
    """
    Open set version for precision score ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives only considering the known class and ``fp`` the number of false positive,
    but also considering the unknown classes

    :param y_true: Correct (observed) values
    :param y_pred: Predicted values
    :param unknown_label: label representing unknown class
    :param labels: possible labels for the classes. If None, considers unique values from ``y_true``, ``y_pred``, ``unknown_label``
    :param return_dict: if `False` return tupples. if `True` returns a dictionary.
    :return: True positive, false negative,  false positive and true negative for each class. ((tps), (fns), (fps), (tns))
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # -- get possible labels forcing "unknown" class to be the first -- #
    if labels is None:
        labels = np.unique([y_true, y_pred])
        labels = np.sort(labels)
        labels = np.array([unknown_label] + [xx for xx in labels if xx != unknown_label])
    # idx_unknown = np.where(labels == unknown_label)
    # idx_unknown = idx_unknown[0][0]

    # -------------------------------------------------------------------- #
    # -- get confusion matrix terms (true positives, false negatives, false positives)
    # -------------------------------------------------------------------- #
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    tps = []
    fns = []
    fps = []
    tns = []
    terms = dict()
    i = 0  # counter starts at zero to skip the "unknown" class
    while i < conf_mat.shape[0] - 1:
        i = i + 1

        # true positives
        i_tp = conf_mat[i, i]
        tps.append(i_tp)

        # sum of false negatives (sum of the line minus the ith element)
        i_fn = np.sum(conf_mat[i, :]) - conf_mat[i, i]
        fns.append(i_fn)

        # sum of false positives (sum of the column minus the ith element)
        i_fp = np.sum(conf_mat[:, i]) - conf_mat[i, i]
        fps.append(i_fp)

        # sum of true negatives. true negative does not consider unknown
        conf_mat_2 = conf_mat.copy()
        conf_mat_2[i, ] = 0
        conf_mat_2[: ,i] = 0
        conf_mat_2[0,] = 0
        conf_mat_2[:, 0] = 0
        i_tn = sum(conf_mat_2.reshape([1,-1])[0])
        tns.append(i_tn)

        terms[labels[i]] = dict(
            tp=i_tp
            , fn=i_fn
            , fp=i_fp
            , tn=i_tn
        )

    tps = tuple(tps)
    fns = tuple(fns)
    fps = tuple(fps)
    tns = tuple(tns)

    if return_dict:
        results = terms
    else:
        results = tps, fns, fps, tns


    return results




# y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, -1]
# y_pred = [0, 1, 1, 1, 1, 1, 1, 0, 0, -1]
# unknown_label = -1

# get_os_conf_mat_terms(y_true, y_pred, unknown_label)
# get_os_conf_mat_terms(
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, -1]
#     , [0, 1, 1, 1, 1, 1, 1, 0, 0, -1]
#     , -1
#     , return_dict=True
# )

