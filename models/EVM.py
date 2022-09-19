# -------------------------------------- #
# --
# -------------------------------------- #

# from .get_os_conf_mat_terms import get_os_conf_mat_terms
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats


class EMV(BaseEstimator, ClassifierMixin):
    """Fits a logistic regression model on tree embeddings.
    """
    def __init__(self
                 , margin_scale = 1  # ?? For some reason this is used to scales the distances in distance matrix
                 , tail_size = 20  # number of observations to use
                 , **kwargs
    ):
        """
        :param margin_scale: For some reason this is used to scales the distances in distance matrix.
        :param tail_size: number of observations to use in the prediction by averaging out their individual predictions
        :param kwargs:
        """

        self.kwargs = kwargs
        self.margin_scale = margin_scale
        self.tail_size = tail_size


    def fit(self, X, y=None):


        self.gbm.fit(X, y)
        X_emb = self.gbm.apply(X).reshape(X.shape[0], -1)
        X_emb = self.bin.fit_transform(X_emb)
        self.lr.fit(X_emb, y)

    def predict(self, X, y=None, with_tree=False):
        if with_tree:
            preds = self.gbm.predict(X)
        else:
            X_emb = self.gbm.apply(X).reshape(X.shape[0], -1)
            X_emb = self.bin.transform(X_emb)
            preds = self.lr.predict(X_emb)
        return preds

    def predict_proba(self, X, y=None, with_tree=False):
        if with_tree:
            preds = self.gbm.predict_proba(X)
        else:
            X_emb = self.gbm.apply(X).reshape(X.shape[0], -1)
            X_emb = self.bin.transform(X_emb)
            preds = self.lr.predict_proba(X_emb)
        return preds


def fit_weibull(
    observation
    , distances
    , margin_scale=1  # ?? For some reason this is used to scale the distances in distance matrix
        ):

    """
    Fit a weibull extreme value machine to ONE observation
    :param observation: an array containing the observation features (vector)
    :param distances: distances between `observation` and observations from other classes.
    :param margin_scale: For some reason this is used to scale the distances in distance matrix.
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

    weibull_pars = dict(
        shape=params[1]
        , scale=params[3])

    # ------------------------- #
    # -- results
    # ------------------------- #
    results = dict(
        weibull_pars=weibull_pars
        , observation=observation
        , distances=distances
    )


