# ---------------------------------------------------------------------- #
# -- Class SKLEARN_EVM
# implements my version of EVM.MultipleEVM
# https://pypi.org/project/EVM/
# ---------------------------------------------------------------------- #

from EVM import MultipleEVM
import scipy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
import ostools

class SKLEARN_EVM(BaseEstimator, ClassifierMixin):
    """
    Fit EVM using https://pypi.org/project/EVM/
    """
    def __init__(self
                 , margin_scale=1
                 , tail_size=20
                 , n_obs_to_fuse=4
                 , confidence_threshold=0.9

                 , cover_threshold = 100
                 , distance_function=scipy.spatial.distance.euclidean
                 , **kwargs
    ):
        """

        :param margin_scale: For some reason this is used to scale the distances in distance matrix.
        :param tail_size: number of observations to use to fit the weibull distribution
        :param n_obs_to_fuse: number of observations to use in the prediction by averaging out their individual predictions
        :param confidence_threshold: Minimum probability of belonging to the acceptance area of the observation
        :param cover_threshold: How much of the original model should not be reduced. The lower `cover_threshold` the fewer observations (with it's weibuls) will remain on the EVM.
        :param kwargs:
        """

        self.kwargs = kwargs
        self.margin_scale = margin_scale
        self.tail_size = tail_size
        self.confidence_threshold = confidence_threshold

        self.cover_threshold = cover_threshold
        self.distance_function = distance_function
        self.distance_multiplier = self.margin_scale

        if n_obs_to_fuse < 1:
            n_obs_to_fuse = None
        self.n_obs_to_fuse = n_obs_to_fuse


    def fit(self, X, y=None):
        # ------------------------------------------------------------------------ #
        # -- make list of features for the given classes to feed EVM.MultipleEVM.train()
        # ------------------------------------------------------------------------ #
        classes = np.unique(y)
        self.classes = classes

        classes_features_list = []
        for class_ in classes:
            aux_features = X[y == class_]
            classes_features_list.append(aux_features)
            del aux_features

        # ------------------------------------------------------------------------ #
        # fit multi EVM
        # ------------------------------------------------------------------------ #
        cover_threshold_ = self.cover_threshold
        if cover_threshold_ < X.shape[0]:
            cover_threshold_ = X.shape[0]
        multi_EVM = MultipleEVM(
            tailsize=self.tail_size
            # , cover_threshold=100 # set to 100 so that it uses all points to create weibull models (Extreme vectors)
            , cover_threshold=cover_threshold_
            # set to 100 so that it uses all points to create weibull models (Extreme vectors)
            , distance_function=self.distance_function
            , distance_multiplier=self.margin_scale
            # I still do not knwo why we should rescale the distances. Easier to compute??
        )

        multi_EVM.train(classes_features_list)

        self.multi_EVM = multi_EVM


    def _predcit_EVM(
            self
            , X
            , return_df=False
            , n_obs_to_fuse = None
    ):
        """
        Auxiliar function to get predictions from EVM.MultipleEVM()
        :param X:
        :param return_df:
        :return:
        """

        # probability of all evms
        evm_probs = self.multi_EVM.probabilities(X)

        obs_preds = []
        for obs in evm_probs:
            class_preds = []
            for class_probs in obs:
                # assert that n_obs_to_fuse is not bigger than available vectors.
                # if (n_obs_to_fuse is None) or (n_obs_to_fuse > class_probs.shape[0]):
                #     n_obs_to_fuse = class_probs.shape[0]

                # get top n_obs_to_fuse
                idx = (-class_probs).argsort()[:n_obs_to_fuse]
                class_preds.append(np.mean(class_probs[idx]))

            class_preds = np.array(class_preds)
            obs_preds.append(class_preds)

        obs_preds = np.array(obs_preds)

        if return_df:
            obs_preds = pd.DataFrame(obs_preds)

        return obs_preds

    def predict_proba(
            self
            , X
            , return_df=False
            , n_obs_to_fuse=None):
        if n_obs_to_fuse is None:
            n_obs_to_fuse = self.n_obs_to_fuse
        # --------------------------------- #
        # predict
        # --------------------------------- #

        predictions_ = self._predcit_EVM(
            X = X
            , return_df=return_df
            , n_obs_to_fuse=n_obs_to_fuse)

        return predictions_

    def predict(
            self
            , X
            , n_obs_to_fuse=None
            , confidence_threshold=None
            , return_df = False
    ):
        """
        """

        if n_obs_to_fuse is None:
            n_obs_to_fuse = self.n_obs_to_fuse

        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        if n_obs_to_fuse is None:
            n_obs_to_fuse = self.n_obs_to_fuse

        # --------------------------------------------- #
        # -- predict class
        # --------------------------------------------- #
        predictions = self.predict_proba(
            X = X
            , n_obs_to_fuse=n_obs_to_fuse
            , return_df=True
        )

        predicted_class = predictions.apply(np.argmax, axis=1)

        # -- compare with threshold
        max_prob = predictions.apply(np.max, axis=1)
        predicted_class[max_prob < confidence_threshold] = -1

        if return_df:
            predicted_class = pd.DataFrame(predicted_class)

        return predicted_class

