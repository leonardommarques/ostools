# -------------------------------------- #
# --
# -------------------------------------- #

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
from scipy.spatial import distance_matrix

sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
from ostools.models.EVM import fit_weibull, predict_weibul

# ---------------------------------------------------------------------------------- #
# -- auxiliary functions
# ---------------------------------------------------------------------------------- #

def get_activations(original_model, X_batch, layer = -2):
    """
    Get the values for the desired layer in the network.
    :param original_model: A `keras.engine.sequential.Sequential`
    :param X_batch: Input values
    :param layer: The index of the desired layer
    :return: activations o the desired layer.
    """
    from keras.models import Model

    intermediate_layer_model = Model(
        inputs=original_model.input
        , outputs=original_model.get_layer(index=layer).output)

    activations = intermediate_layer_model.predict(X_batch)

    return activations


# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #
# the class itself



class OpenMax(BaseEstimator, ClassifierMixin):
    def __init__(self
                 , trained_model
                 , tail_size = 10
                 , classes_to_revise = -1):
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

        """

        self.trained_model = trained_model
        self.tail_size = tail_size
        self.classes_to_revise = classes_to_revise

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
        av_df = get_activations(original_model=trained_model
                              , X_batch=X
                              , layer=-2)
        av_df = pd.DataFrame(av_df)
        av_df['y'] = y
        av_df['pred'] = predictions

        # ----------------------------- #
        # -- Mean activation vectors
        # ----------------------------- #
        # -- get correct predictions -- #
        idx_correct_predictions = av_df['pred'] == av_df['y']
        correct_df = av_df[idx_correct_predictions]
        # correct_df.query('pred == y').query('y == 0').shape
        # correct_df.query('pred == y').query('y == 1').shape

        # -- MAV -- #
        MAV_df = correct_df.drop(columns=['pred']).groupby('y').mean()
        MAV_df = MAV_df.reset_index()

        # ----------------------------- #
        # -- weibulls
        # ----------------------------- #

        labels_ = MAV_df['y'].unique()
        weibulls_dct = {}
        for label_ in labels_:
            # label_ = labels_[0]

            # -- x of the given label -- #
            x_ = correct_df.query('y == @label_')
            x_ = x_.drop(columns = ['y', 'pred'])

            # -- MAV of the given label -- #
            mav_ = MAV_df.query('y == @label_').drop(columns = ['y'])

            # -- Get longest distances to MAV -- #
            dists_ = distance_matrix(mav_, x_)
            dists_ = dists_[0]
            dists_ = np.sort(dists_)
            longest_dists = dists_[-tail_size:]

            # -- fit weibull -- #
            weibull_ = fit_weibull(
                observation=mav_
                , distances=longest_dists)
            weibull_['label'] = label_

            weibulls_dct[label_] = weibull_

        self.weibulls = weibulls_dct
        self.mean_activation_vectors = MAV_df


    def predict_proba(
            self
            , X):
        """

        :param X: Features to predict
        :return:
        """

        trained_model = self.trained_model
        weibulls_dct = self.weibulls
        classes_to_revise = self.classes_to_revise

        if classes_to_revise == -1:
            ALPHA = len(weibulls_dct)
        else:
            ALPHA = classes_to_revise

        # -------------------------------------- #
        # -- Activation vectors
        # -------------------------------------- #
        av_df = get_activations(original_model=trained_model
                                , X_batch=X
                                , layer=-2)
        av_df = pd.DataFrame(av_df)
        av_df.columns = [f"__activation_{i}" for i in av_df.columns]
        av_cols = [i for i in av_df.columns]

        # -------------------------------------- #
        # -- Distances
        # -------------------------------------- #
        dists = {}
        for label_, pars_ in weibulls_dct.items():
            dists_ = distance_matrix(av_df, pars_['observation'])
            dists[label_] = dists_

        for label_ in dists.keys():
            av_df[f'__dist_to_{label_}'] = dists[label_]

        # -------------------------------- #
        # -- Weibull cdf
        # -------------------------------- #

        labels_ = list(dists.keys())
        for label_ in labels_:
            # label_ = 0
            # label_ = 1
            # -- get the weibull pdf for the given class -- #
            pdf_ith_class_ = predict_weibul(
                av_df[f'__dist_to_{label_}']
                , weibull_pars = weibulls_dct[label_]['weibull_pars']['params']
                , lower_tail = True
            )

            av_df[f'__pdf_ith_class_{label_}'] = pdf_ith_class_

        # ---------------------------------------------- #
        # -- Weights it according to activation order -- #
        # ---------------------------------------------- #
        weight_activation_order = av_df[av_cols].apply(lambda xx: np.argsort(xx) + 1, 1)
        weight_activation_order.columns = [i.replace('__activation_', '__weight_') for i in  weight_activation_order.columns]
        weight_activation_order = weight_activation_order.apply(lambda xx: (ALPHA-xx)/ALPHA, 1)

        for label_ in labels_:
            av_df[f"__weight_{label_}"] = 1 - weight_activation_order[f"__weight_{label_}"] * av_df[f"__pdf_ith_class_{label_}"]

        # ---------------------------------------------- #
        # -- Revise vectors, Line 5
        # ---------------------------------------------- #
        for label_ in labels_:
            av_df[f"__revised_{label_}"] = av_df[f"__activation_{label_}"]*(av_df[f"__weight_{label_}"])
            # av_df[f"__weight_{label_}"] = 1 - weight_activation_order[f"__weight_{label_}"] * av_df[f"__pdf_ith_class_{label_}"]

        # ---------------------------------------------- #
        # -- Define V0, line 6
        # ---------------------------------------------- #
        for label_ in labels_:
            av_df[f"__v_zero_{label_}"] = av_df[f"__revised_{label_}"] * (1-av_df[f"__weight_{label_}"])

        # ---------------------------------------------- #
        # -- Openmax
        # ---------------------------------------------- #
        for label_ in labels_:
            # scipy.special.softmax()
            # av_df[[f"__v_zero_{label_}" for label_ in labels_]]

            av_df[[f"__v_zero_{label_}" for label_ in labels_]].apply(scipy.special.softmax, 1)



av_df['__pdf_ith_class_1'].unique()
av_df['__weight_1'].unique()
av_df['__pdf_ith_class_1']

# ---------------------------------------- #
# --
# ---------------------------------------- #
# data and model: /Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/misc/studies/activation Layers Softmax vs Openmax.py
model

