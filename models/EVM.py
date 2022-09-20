# -------------------------------------- #
# --
# -------------------------------------- #

# from .get_os_conf_mat_terms import get_os_conf_mat_terms
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
from scipy.spatial import distance_matrix

# ----------------------------------------- #
# -- auxiliary functions
# ----------------------------------------- #

def fit_weibull(
        observation
        , distances
        , return_distances=False
        ):

    """
    Fit a weibull extreme value machine to ONE observation
    :param observation: an array containing the observation features (vector)
    :param distances: distances between `observation` and observations from other classes.
    :param margin_scale: For some reason this is used to scale the distances in distance matrix.
    :param return_distances: if `True` return the parameter `distances` in the results.
    :return: a weibull model for the extreme value of the observation
    """
    # observation = [-1.91979003066328, -0.803488581854428]
    # distances = [0.255611000126613, 0.537614268131727, 0.722064217565286, 1.44670889923755,1.57965222614633, 1.75658662230802, 2.08236734718123, 2.306139687319,
    #   2.3170316764444, 2.35437078692245, 2.63403964040791, 2.78613588523478,3.05555306091183, 3.29969688074233, 3.3184464431157, 3.34687210363261,
    #   3.43187192195839, 3.76010038148846, 3.77427932967755, 3.90003670677621]

    # from scipy import stats

    # ------------------------- #
    # -- fit weibull -- #
    # ------------------------- #

    params = stats.exponweib.fit(distances, floc=0, f0=1)
    exp1, k1, loc1, lam1 = params
    shape_ = k1
    scale_ = lam1

    weibull_pars = dict(
        shape=shape_
        , scale=scale_
        , params = params
    )

    # ------------------------- #
    # -- results
    # ------------------------- #
    results = dict(
        weibull_pars=weibull_pars
        , observation=observation
    )
    if return_distances:
        results['distances']=distances


    return results



def predict_weibul(
        x
        , weibull_pars
        , lower_tail = True
):
    """
    Make weibull predictions with given parameters
    :param x: quantiles
    :param weibull_pars: parameters for the weibull distribution. if lenth=2, then considers exp1=1, and loc1=0
    :return:
    """
    if len(weibull_pars) == 2:
        exp1, k1, loc1, lam1 = 1, weibull_pars[1], 0, weibull_pars[3]
    else:
        exp1, k1, loc1, lam1 = weibull_pars

    result = stats.exponweib.cdf(x, exp1, k1, loc1, lam1)

    if lower_tail:
        result = 1-result

    return result


def predict_from_weibulls_dict_(
        X
        , weibulls_dict
        , n_obs_to_fuse=4
        , return_dict = True):
    """
    Predict form a `weibulls_dict`.
    :param X: "test" observation to predict to. Must be a `(1,n_col)` array (n_col being the number of features)
    :param weibulls_dict: `EVM.weibulls_dict` resulting from `EVM.fit()`
    :param n_obs_to_fuse: number of observations to use in the prediction by averaging out their individual predictions
    :param return_dict: if True, returns a dictionary. If false, return a list.
    :return:
    """
    # X = X_pred
    # X = X_pred[0, ]
    # X = X_pred[4:5, ]
    assert X.shape[0] == 1, 'X is not an array fo shape (1, nnumber of features)'

    # ---------------------------------------- #
    # -- get probability for each class
    # ---------------------------------------- #
    probs_by_class = {i: [] for i in weibulls_dict.keys()}  # to save the probabilities predicitons for each class
    i_labels = -1
    while i_labels < len(weibulls_dict) - 1:
        i_labels = i_labels + 1
        str_label = list(weibulls_dict.keys())[i_labels]

        for weibull in weibulls_dict[str_label]:
            # weibull = weibulls_dict[str_label][0]
            distance = distance_matrix(weibull['observation'].reshape(1, -1), X.reshape(1, -1))

            prob_prediction_ = predict_weibul(
                x=distance
                , weibull_pars=weibull['weibull_pars']['params']
            )

            assert len(prob_prediction_) == 1
            prob_prediction_ = prob_prediction_[0][0]
            probs_by_class[str_label].append(prob_prediction_)
            del prob_prediction_


    # --------------------------------------------------------------- #
    # -- get top predictions to average out (n_obs_to_fuse) -- #
    # --------------------------------------------------------------- #
    for key_, value_ in probs_by_class.items():
        value_.sort(reverse=True)
        if n_obs_to_fuse is not None:
            value_ = value_[:n_obs_to_fuse]
        probs_by_class[key_] = value_

    predictions = {key_: np.mean(value_) for key_, value_ in probs_by_class.items()}
    if not return_dict:
        predictions = list(predictions.values())

    # ------------------ #
    # -- results
    # ------------------ #
    return predictions


# ----------------------------------------- #
# ----------------------------------------- #


class EVM(BaseEstimator, ClassifierMixin):
    """Fits a logistic regression model on tree embeddings.
    """
    def __init__(self
                 , margin_scale=1
                 , tail_size=20
                 , n_obs_to_fuse=4
                 , confidence_threshold=0.9
                 , **kwargs
    ):
        """

        :param margin_scale: For some reason this is used to scale the distances in distance matrix.
        :param tail_size: number of observations to use to fit the weibull distribution
        :param n_obs_to_fuse: number of observations to use in the prediction by averaging out their individual predictions
        :param confidence_threshold: Minimum probability of belonging to the acceptance area of the observation
        :param kwargs:
        """

        self.kwargs = kwargs
        self.margin_scale = margin_scale
        self.tail_size = tail_size
        self.confidence_threshold = confidence_threshold

        if n_obs_to_fuse < 1:
            n_obs_to_fuse = None
        self.n_obs_to_fuse = n_obs_to_fuse


    def fit(self, X, y=None):
        """

        :param X: Features
        :param y: observed classes
        :return:
        """

        tail_size = self.tail_size
        margin_scale = self.margin_scale

        # -------------------------- #
        # -------------------------- #
        self.labels = np.unique(y)

        # -------------------------- #
        # -- Get all distances
        # -------------------------- #

        dist_mat = distance_matrix(X, X)

        # -------------------------- #
        # -- Fit one weibull for each observation
        # -------------------------- #
        # weibulls = []
        weibulls_dict = {i: [] for i in np.unique(y)}
        i_observation = -1
        while i_observation < y.shape[0]-1:
            i_observation = i_observation + 1

            # -- observation features and label -- #
            obs_features = X[i_observation, ]
            obs_label = y[i_observation]

            # -- distances to other classes -- #
            obs_distances = dist_mat[i_observation, y != obs_label]
            obs_distances = np.sort(obs_distances)
            if tail_size is not None:
                # -- handle when there are too few observations for the tail size
                if tail_size > len(obs_distances):
                    obs_distances = obs_distances[:len(obs_distances)]
                else:
                    obs_distances = obs_distances[:tail_size]

            obs_distances = obs_distances*margin_scale

            # -- fit weibull -- #
            obs_weibull = fit_weibull(
                observation=obs_features
                , distances=obs_distances)
            obs_weibull['label'] = obs_label
            weibulls_dict[obs_label].append(obs_weibull)

        # -- Save results -- #
        self.weibulls_dict = weibulls_dict


    def predict_proba(
            self
            , X
            , return_dict = True
            , n_obs_to_fuse = None
    ):
        """
        Predictions using the weibulls
        :param X: "test" observations to predict to.
        :param return_dict: if True, returns a dictionary. If false, return a list.
        :param n_obs_to_fuse: number of observations to use in the prediction by averaging out their individual predictions
        :param confidence_threshold: Minimum probability of belonging to the acceptance area of the observation
        :return:
        """

        # -- get hyper parameters -- #
        weibulls_dict = self.weibulls_dict

        if n_obs_to_fuse is None:
            n_obs_to_fuse = self.n_obs_to_fuse

        # --------------------------- #
        # -- make predicitons
        # for each test observation
        # --------------------------- #

        predictions = []
        i_obs = -1
        while i_obs < X.shape[0]-1:
            i_obs = i_obs + 1
            X_aux = X[i_obs, ]
            X_aux = X_aux.reshape(1, -1)

            pred_ = predict_from_weibulls_dict_(
                X_aux
                , weibulls_dict=weibulls_dict
                , return_dict = return_dict
                , n_obs_to_fuse = n_obs_to_fuse
            )

            predictions.append(pred_)

        if not return_dict:
            # -- transform to array -- #
            # predictions_array = [list(i.values()) for i in predictions]
            predictions = np.array(predictions)


        return predictions

    def predict(
            self
            , X
            , n_obs_to_fuse = None
            , confidence_threshold = None):

        labels = self.labels
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # ------------------------------- #
        # -- get probability predictions -- #
        # ------------------------------- #
        predictions = self.predict_proba(X
            , return_dict=False
            , n_obs_to_fuse=None
            )

        # ------------------------------- #
        # -- predict label
        # ------------------------------- #

        proba_most_likely_label = np.apply_along_axis(lambda xx: np.max(xx), 1, predictions)
        predicted_label_idx = np.apply_along_axis(lambda xx: np.where(xx == np.max(xx))[0], 1, predictions)
        predicted_label_ = np.apply_along_axis(lambda xx: labels[xx], 0, predicted_label_idx)

        # -- check if prediction of for unknown class (predicted value < confidence_threshold
        predicted_label_[proba_most_likely_label < confidence_threshold] = -1
        predicted_label_ = predicted_label_.reshape(-1, 1)

        # ----------------------------- #
        # -- result
        # ----------------------------- #
        return predicted_label_




