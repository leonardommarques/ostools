# -------------------------------------- #
# --
# -------------------------------------- #

import numpy as np
from scipy.stats import weibull_min


def get_activations(
        original_model
        , x_batch
        , layer=-2
):
    """
    Get the values for the desired layer in the network.
    :param original_model:
    :param X_batch: Input values
    :param layer: The index of the desired layer
    :return: activations o the desired layer.
    """
    from keras.models import Model

    intermediate_layer_model = Model(
        inputs=original_model.input
        , outputs=original_model.get_layer(index=layer).output)

    activations = intermediate_layer_model.predict(x_batch)

    return activations

# -------------------------------------------------------------- #
# -------------------------------------------------------------- #
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class WeibullScipy(BaseEstimator, RegressorMixin):

    def __init__(self):
        pass

    def fit(self, X):
        """
        Fit a 3 parametric weibull distribution.
        :param X: a 1-dimensional numpy array, or `list`
        :return:
        """
        weibull_pars = weibull_min.fit(X)
        shape, loc, scale = weibull_pars

        weibull_pars_dict = dict(
            shape = shape, loc = loc, scale = scale
        )

        self.pars = weibull_pars
        self.pars_dict = weibull_pars_dict

    def predict(self, X):
        """
        CDF of the fitted weibull.
        :param X: array of observations, x values to predict to.
        :return: cumulative distribution estimate of the fitted weibull.
        """
        # k/shape, loc/loc_, lam/scale
        # pars_dict = weibull_pars_dict
        weibull_pars_dict = self.pars_dict

        preds = weibull_min.cdf(
            X
            , c=weibull_pars_dict['shape']
            , loc=weibull_pars_dict['loc']
            , scale=weibull_pars_dict['scale']
        )

        return preds
    def cdf(self, X):
        return self.predict(X)




# from scipy.stats import weibull_min
# aux_sample = weibull_min.rvs(
#     k, loc=loc_
#     , scale=lam, size=n
# )
# # ------------------------- #
# # -- scipy
# import time
# start_time = time.time()
# pars = weibull_min.fit(aux_sample) # k/shape, loc/loc_, lam/scale

# -------------------------------------------------------------- #
# -- Calculate alphas
# -------------------------------------------------------------- #
def compute_alpha_weights(activation_vector):
    """
    Calculate the alpha weights used in Algorithm 2 (https://arxiv.org/pdf/1511.06233.pdf)
    :param activation_vector:
    :return:
    """

    activation_ranks = activation_vector.argsort().ravel()[::-1]
    alpharank = len(activation_ranks)
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]

    # order the alphas to the corresponding activation. The highest activation has the highest alpha.
    ranked_alpha = np.zeros(alpharank)
    for i in range(len(alpha_weights)):
        ranked_alpha[activation_ranks[i]] = alpha_weights[i]

    return ranked_alpha


