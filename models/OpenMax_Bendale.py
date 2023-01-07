# -------------------------------------- #
# --
# -------------------------------------- #

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import scipy

sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')

from ostools.models.EVM import fit_weibull, predict_weibul
from ostools.functions import compute_alpha_weights, get_activations

# ---------------------------------------------------------------------------------- #
# -- auxiliary functions
# ---------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #
# the class itself

class OpenMax_Bendale(BaseEstimator, ClassifierMixin):
    def __init__(self
                 , trained_model
                 , tail_size=10
                 , classes_to_revise=-1
                 , distance_function=scipy.spatial.distance.euclidean
                 ):
        """
        OpenMax model presented in Towards Open Set Deep Network (https://arxiv.org/pdf/1511.06233.pdf)

        :param trained_model: a `keras.engine.sequential.Sequential` having the last layer being a
        `keras.layers.core.activation.Activation` and the second last being a `keras.layers.core.dense.Dense`.
        WRONG way to specify the last layer:
                `model.add(Dense(units = ..., activation='softmax'))`
        RIGHT way to specify the last layer:
                `model.add(Dense(units = ..., activation='linear'))`
                `model.add(Activation('softmax'))`

        :param tail_size: Number of observations to use when estimating the rejection boundaries.
        :param classes_to_revise: Number of classes to use when predicting. (Algorithm 2. Parameter alpha)
        :param distance_function: Function to calculate the distance between MAV and the observations

        """

        self.trained_model = trained_model
        self.tail_size = tail_size
        self.classes_to_revise = classes_to_revise
        self.distance_function = distance_function

    def fit(
            self
            , X
            , y=None
            , predictions=None
    ):
        """
        Fit weibulls

        :param X: Features
        :param y: Observed value
        :param predictions: predicted values. If, `model.predict(X)` are used.
        :return:
        """

        trained_model = self.trained_model
        tail_size = self.tail_size
        distance_function = self.distance_function

        # ----------------------------- #
        # -- sanitization
        # ----------------------------- #
        if predictions is None:
            predictions = trained_model.predict(X)
            predictions = np.apply_along_axis(np.argmax, 1, predictions)

        # -- enforce y to be one column -- #
        if len(y.shape) > 1:
            y = np.apply_along_axis(np.argmax, 1, y)

        # ----------------------------- #
        # -- Activation vectors
        # ----------------------------- #
        activation_vectors = get_activations(
            original_model=trained_model
            , x_batch=X
            , layer=-2)

        labels_ = np.unique(y)
        self.labels = labels_

        # ---------------------------------------------------------- #
        # -- Mean activation vectors
        # ---------------------------------------------------------- #
        # -- get correct predictions and theis activations -- #
        idx_correct_predictions = predictions == y
        activation_vectors_correct = activation_vectors[idx_correct_predictions].copy()
        y_correct = y[idx_correct_predictions].copy()

        # -- MAV -- #
        activation_vectors_correct_dict = {key_: activation_vectors_correct[y_correct == key_] for key_ in labels_}
        weibull_model_pars = {}
        for key_, value_ in activation_vectors_correct_dict.items():
            aux_mav = np.apply_along_axis(np.mean, 0, value_)

            weibull_model_pars[key_] = {'mav': aux_mav}

        # ----------------------------- #
        # -- Distances -- #
        # ----------------------------- #

        for key_, value_ in weibull_model_pars.items():
            aux_x = activation_vectors_correct_dict[key_]
            aux_mav = weibull_model_pars[key_]['mav']

            aux_distances = np.apply_along_axis(
                lambda xx: distance_function(xx, aux_mav)
                , 1
                , aux_x
            )

            weibull_model_pars[key_]['distances'] = aux_distances
            del aux_x, aux_mav, aux_distances


        # ----------------------------- #
        # -- weibulls
        # ----------------------------- #

        for key_, value_ in weibull_model_pars.items():
            aux_distances = weibull_model_pars[key_]['distances'].copy()
            aux_distances = np.sort(aux_distances)
            shortest_distances = aux_distances[-tail_size: ]

            aux_weibull_model = WeibullScipy()
            aux_weibull_model.fit(shortest_distances)

            weibull_model_pars[key_]['weibull_model'] = aux_weibull_model

            del aux_weibull_model, aux_distances, shortest_distances

        self.weibulls = weibull_model_pars


    def predict_proba(
            self
            , X):
        """
        A dataFrame with the predictions
        :param X: Features to predict
        :return:
        """

        trained_model = self.trained_model
        weibulls_dct = self.weibulls
        classes_to_revise = self.classes_to_revise
        distance_function = self.distance_function

        if classes_to_revise == -1:
            ALPHA = len(weibulls_dct)
        else:
            ALPHA = classes_to_revise

        # -------------------------------------- #
        # -- Activation vectors
        # -------------------------------------- #
        activation_vectors = get_activations(
            original_model=trained_model
            , x_batch=X
            , layer=-2)

        # -------------------------------------- #
        # -- CDFs
        # -------------------------------------- #

        cdfs = {}
        for key_, value in weibulls_dct.items():

            aux_model = weibulls_dct[key_]['weibull_model']
            aux_mav = weibulls_dct[key_]['mav']

            aux_distances = np.apply_along_axis(
                lambda xx: distance_function(xx, aux_mav)
                , 1
                , activation_vectors
            )

            aux_cdf = aux_model.predict(aux_distances)
            cdfs[key_] = aux_cdf
            del aux_cdf, aux_distances, aux_model, aux_mav

        cdfs = pd.DataFrame(cdfs)

        # ----------------------------------- #
        # -- alphas
        # ----------------------------------- #
        alphas = np.apply_along_axis(compute_alpha_weights, 1, cdfs)
        alphas = alphas.reshape(alphas.shape[0], alphas.shape[2])

        # ----------------------------------- #
        # -- revised vector
        # ----------------------------------- #
        revised_known = activation_vectors * (1- alphas * cdfs)
        revised_unknown = activation_vectors - revised_known

        activation_vectors * (1 - alphas * cdfs)

        calculated_openMax = []
        i = -1
        while i < activation_vectors.shape[0] - 1:
            i = i + 1
            aux_calculated_openMax = computeOpenMaxProbability(
                np.array(list([revised_known.values[i]]))
                , np.array(list([revised_unknown.values[i]]))
            )

            calculated_openMax.append(aux_calculated_openMax)

        calculated_openMax_df = pd.DataFrame(np.array(calculated_openMax))
        calculated_openMax_df.columns = list(labels_) + [-1]
        calculated_openMax_df.columns = [str(i) for i in calculated_openMax_df.columns]

        return calculated_openMax_df

# ---------------------------------------- #
# --
# ---------------------------------------- #
# data and model: /Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/misc/studies/activation Layers Softmax vs Openmax.py
model

