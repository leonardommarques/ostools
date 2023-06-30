import os
import sys
import numpy as np
import pandas as pd
# import random
# import math
# import pickle
# from tqdm import tqdm
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Input, Conv1D, MaxPooling1D, \
# Dropout, BatchNormalization, Activation, Concatenate, Masking, GlobalAveragePooling1D, \
# Conv1DTranspose, Reshape, PReLU
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.regularizers import l2
# from keras.utils.np_utils import to_categorical
# from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
# from tslearn.datasets import UCR_UEA_datasets
# from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
# from sktime.utils.data_io import load_from_tsfile_to_dataframe
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.utils import class_weight
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as stats
from scipy.stats import weibull_min
# from scipy.special import softmax
# import scipy.spatial.distance as distance
# import scipy.optimize as optimize
import warnings
warnings.filterwarnings("ignore")

# ---------------------------- #
# -- OpenMax
# ---------------------------- #

# ---------------------------------------- #
# --
# ---------------------------------------- #

def calc_distances(activations, mav):
    """
    Returns the distances between the activations of correctly classified samples and MAV of the given class
    """
    distances = np.empty(len(activations))
    for i in range(len(activations)):
        distances[i] = np.linalg.norm(activations[i] - mav)
    return distances


# ---------------------------------------- #
# --
# ---------------------------------------- #

def fit_weibull(distances_all, n_classes):
    """
    Returns one Weibull model (set of parameters) for each class
    """
    weibull_models = []
    for i in range(n_classes):
        shape, loc, scale = weibull_min.fit(distances_all[i])
        weibull_models.append([shape, loc, scale])

    return weibull_models



# ---------------------------------------- #
# --
# ---------------------------------------- #

def predict_openmax(x, mav_list, weibull_models, alpha = None):
    """

    :param x: Activation vector
    :param mav_list:
    :param weibull_models:
    :param alpha:
    :return:
    """
    n_classes = x.shape[1]
    openmax_scores = np.zeros(shape=(x.shape[0], x.shape[1] + 1))
    alpha = alpha # number of top classes to revise
    if alpha is None:
        alpha = x.shape[1]
    # for each sample in x
    for n in range(len(x)):
        # n = 0
        av = x[n, ]
        ranks = np.argsort(av)[::-1]
        y_hats = []
        y_hat_unknown = 0 # score for the unknown class
        # for each known class
        for i in range(n_classes):
            # i = 0
            dist = np.linalg.norm(av - mav_list[i])
            shape, loc, scale = weibull_models[i]
            weibull_score = weibull_min.cdf(dist, shape, loc, scale)
            r_i = np.where(ranks==i)[0][0]
            R_alpha = max(0, ((alpha - r_i) / alpha))
            w_i = 1 - R_alpha * weibull_score # probability of belonging to the given class
            # w_i = 1 - weibull_score
            y_hat_i = av[i] * w_i  # openmax score for the given class
            y_hats.append(y_hat_i)
            y_hat_unknown += av[i] * (1 - w_i)

        # append the unknown class score
        y_hats.append(y_hat_unknown)
        openmax_scores[n] = softmax(y_hats)

    return openmax_scores

# ---------------------------------------- #
# --
# ---------------------------------------- #


class OpenMax():
    def __int__(
            self
            , ratio_top_distances = 0.1 # ratio of top distances to use for weibull models.
                ):
        pass

    def fit(self, X_avs, y, y_hat):
        """
        Fit OpenMax layer for given activation vectors `X_avs`, targets `y` and predictions `y_hat` (when provided).
        :param X_avs:
        :param y: Target.
        :param y_hat: Predictions. They are used to get the correct classified data.
        :return:
        """

        # ----------------------------- #
        # -- get correct predictions
        # ----------------------------- #
        idx_correct = y == y_hat
        correct_X_avs = X_avs[idx_correct]
        correct_y = y[idx_correct]

        # ----------------------------- #
        # -- split by target
        # ----------------------------- #
        correct_classifications_groups = {}
        for group in np.unique(correct_y):
            idx_group = correct_y == group
            correct_classifications_groups[str(group)] = correct_X_avs[idx_group]
            del idx_group

        # ----------------------------- #
        # -- OpenMax attributes
        # ----------------------------- #
        mavs_dict = {}
        distances_dict = {}
        top_distances_all = {}
        ratio_top_distances = .1
        weibull_models = {}
        for key_, value_ in correct_classifications_groups.items():
            mav = np.mean(value_, axis=0)  # MAV for the given class
            distance_ = calc_distances(value_, mav)
            mavs_dict[key_] = mav
            distances_dict[key_] = (distance_)
            top_distances_all[key_] = np.sort(distance_)[int(-len(distance_) * ratio_top_distances):]

            # = shape, loc, scale = weibull_min.fit(distances_all[i])
            weibull_pars = weibull_min.fit(top_distances_all[key_])
            weibull_models[key_] = weibull_pars

        # ------------------------- #
        # == Save attributes
        # ------------------------- #
        self.mavs_dict = mavs_dict
        self.weibull_models = weibull_models

    def predict(self, X_avs, alpha = None):
        """
        Predict using OpenMax
        :param X_avs: Activation vectors
        :param alpha: Number of top classes to revise.
        :return:
        """

        predictions_ = predict_openmax(
            x=X_avs
            , mav_list = list(mavs_dict.values())
            , weibull_models = list(weibull_models.values())
            , alpha=alpha
        )

        return predictions_