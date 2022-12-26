################################################################################
# Softmax VS OpenMax
# I try to implement OpenMax
################################################################################

### Hyper parameters
tail_size = 20

import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import plotnine as gg

import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
# from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
# from ostools.models import EVM
# from datetime import datetime
from ostools.functions import WeibullScipy
from ostools.functions import get_activations

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Softmax
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import scipy
import libmr


# ------------------------------------------------------------ #
# -- data
# ------------------------------------------------------------ #
# -- create example data
np.random.seed(2956)
n = 50

da1 = pd.DataFrame({
    'x1': np.random.normal(loc=1, scale=2, size=n)
    , 'x2': np.random.normal(loc=1, scale=2, size=n)
    , 'y':'0'
})

da2 = pd.DataFrame({
    'x1': np.random.normal(loc=20, scale=2, size=n)
    , 'x2': np.random.normal(loc=30, scale=2, size=n)
    , 'y':'1'
})

db1 = pd.DataFrame({
    'x1': np.random.normal(loc=30, scale=2, size=n)
    , 'x2': np.random.normal(loc=2, scale=2, size=n)
    , 'y':'-1'
})

total_da = pd.concat([da1, da2, db1])
known_da = pd.concat([da1, da2])
del da1, da2

# (
#         gg.ggplot(
#     total_da
#     , gg.aes(x = 'x1', y = 'x2', fill = 'y')
#         ) +
#         gg.geom_point(size = 4)
# )


# ------------------- #
# -- splits -- #
# ------------------- #
train_da, test_da = train_test_split(known_da, test_size=0.2, random_state=876357463)
train_da, val_da = train_test_split(train_da, test_size=0.2, random_state=876357463)

# -- add unknown data -- #
test_da = pd.concat([test_da, db1])

[a.shape[0] for a in [train_da, val_da, test_da]]


train_da['split'] = 'train'
val_da['split'] = 'val'
test_da['split'] = 'test'

# -- X -- #
X_train = train_da[['x1', 'x2']].values
X_val = val_da[['x1', 'x2']].values
X_test = test_da[['x1', 'x2']].values

# -- Y -- #
y_train = train_da['y'].values
Y_train = pd.get_dummies(y_train)

y_val = val_da['y'].values
Y_val = pd.get_dummies(y_val)

y_test = test_da['y'].values
Y_test = pd.get_dummies(y_test)

# --------- #
# scale
# --------- #
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------ #
# -- model
# ------------------------------------------------------------ #

model = Sequential()

model.add(Dense(
    input_shape = (X_train.shape[1], )
    , units = 10
    , activation='sigmoid'
                ))

model.add(Dense(
    units = 5
    , activation='sigmoid'
                ))

# ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - #
# !!!!!!!!!!!!! It is imperative that the two last layers are a linear and a softmax. So that we can get the
# activation vectors of the penultimate layer

model.add(Dense(
    units = X_train.shape[1]
    , activation='linear'
                ))

model.add(
    Activation('softmax'
          , name='softMax_activation')
)


model.summary()

# ------------------- #
# -- compile
# ------------------- #
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01)
    , loss='binary_crossentropy'
    , metrics=['accuracy']
)

# model.fit(X_train, Y_train, epochs=20)
model.fit(X_train, Y_train, epochs=10)
model.evaluate(X_train, Y_train)
model.evaluate(X_val, Y_val)
# model.evaluate(X_test, Y_test)

# ------------------------------------------- #
# -- Activations and predicitons
# ------------------------------------------- #
X_batch = X_train[:5].copy()

# def get_activations(original_model, X_batch, layer = -2):
#     """
#     Get the values for the desired layer in the network.
#     :param original_model:
#     :param X_batch: Input values
#     :param layer: The index of the desired layer
#     :return: activations o the desired layer.
#     """
#     from keras.models import Model
#
#     intermediate_layer_model = Model(
#         inputs=original_model.input
#         , outputs=original_model.get_layer(index=layer).output)
#
#     activations = intermediate_layer_model.predict(X_batch)
#
#     return activations


# activation vectors
av = get_activations(model, x_batch=X_batch)
# softmax on the activation vetors
np.apply_along_axis(lambda x: [np.exp(i)/sum(np.exp(x)) for i in x], 1, av)
# nn predictions (equal to softmax o the activation vector)
model.predict(X_batch)

# ------------------------------------------- #
# ------------------------------------------- #
# -- Predictions and Activation Vectors
# ------------------------------------------- #
# ------------------------------------------- #

total_da = pd.concat([
    train_da.reset_index(drop=True)
    , val_da.reset_index(drop=True)
    , test_da.reset_index(drop=True)]
    , axis = 0, ignore_index=True)

preds = model.predict(total_da[['x1', 'x2']].values)
total_preds = pd.DataFrame(preds)
total_preds.columns = [f"pred_{i}" for i in range(len(total_preds.columns))]
classes_labels = train_da['y'].unique()
classes_labels.sort()

total_preds['predicted_class'] = total_preds[total_preds.columns].apply(lambda x: np.argmax(x), axis=1)
total_preds['predicted_class'] = classes_labels[total_preds['predicted_class']]
total_da['predicted_class'] = total_preds['predicted_class']

avs = get_activations(model, x_batch=total_da[['x1', 'x2']].values)
avs_df = pd.DataFrame(avs)
avs_df.columns = [f"activation_vector_{i}" for i in range(len(avs_df.columns))]

math_da = pd.concat(
    [
    total_da.copy().reset_index(drop=True)
    , avs_df.reset_index(drop=True)
    ]
    , axis = 1
)
math_da = pd.concat([math_da, total_preds], axis=1)
math_da.head(3)

math_da.to_csv("/Users/leonardomarques/Downloads/total_da_2.csv", index=False)
# Honestly I don't know why I made this dataframe. I think I need professional help and vacations.

# - predictions
for i in range(preds.shape[1]):
    total_da['proba_' + str(i)] = preds[:, i]

# -- AVs
for i in range(preds.shape[1]):
    total_da['av_' + str(i)] = avs[:, i]


# (
#         gg.ggplot(
#     total_da
#     , gg.aes(x = 'proba_0', y = 'proba_1', fill = 'y')
#         ) +
#         gg.geom_point(size = 4)
# )

(
        gg.ggplot(
    total_da
    , gg.aes(x ='av_0'
             , y='av_1'
             , fill='y')
        ) +
        gg.geom_point(size = 4) +
        gg.facet_wrap('~split')
)

# ------------------------------------------- #
# ------------------------------------------- #
# -- Mean activation vectors
# ------------------------------------------- #
# ------------------------------------------- #

# -- get mean activation vectors
## PAPER page 5: The model µi is computed using the images associated with category i, images that were classified correctly (top-1) during training process.
correctly_predicted_da = total_da[total_da['predicted_class'] == total_da['y']]
mavs_da = correctly_predicted_da.query('split == "train"')[['av_0', 'av_1', 'y']].groupby('y').mean().reset_index()

# (
#         gg.ggplot(
#     total_da.query('split == "train"')
#     , gg.aes(x ='av_0'
#              , y='av_1'
#              , fill='y')
#         ) +
#         gg.geom_point(size = 4) +
#         gg.geom_point(
#             data = mavs_da
#             , mapping=gg.aes(x ='av_0'
#                      , y='av_1'
#                      , fill='y')
#             , size = 10
#         )
# )

# -------------------------------------- #
# -- distance to MAV
# -------------------------------------- #
import libmr


i = -1
while i < len(mavs_da['y'])-1:
    i = i+1
    i_class = mavs_da['y'][i]

    # aux_train_da = total_da.query('y == @i_class')
    i_mav = mavs_da.query('y == @i_class').drop(columns = ['y'])

    dist_to_ith_mean = total_da[['av_0', 'av_1']].apply(
        lambda xx: scipy.spatial.distance.euclidean(
            xx
            , i_mav[['av_0', 'av_1']].values
        )
        , axis = 1
    )

    total_da['dist_to_mean_'+i_class] = dist_to_ith_mean


total_da['dist_to_class_mean'] = total_da[[aux for aux in total_da.columns if 'dist_to_mean_' in aux]].apply(min, axis=1)

# ------------------------------------------------ #
# ------------------------------------------------ #
# -- weibull models and CDF
# on the high distances
# ------------------------------------------------ #
# ------------------------------------------------ #
weibull_models = mavs_da.copy()
weibull_models['weibull_model'] = 'dsdsd'
i = -1
while i < len(mavs_da['y'])-1:
    i = i+1
    i_class = mavs_da['y'][i]

    # n highest distances from observations of i-th class and ith mean activation vector
    ith_distances = total_da[total_da['y'] == i_class]['dist_to_class_mean'].values
    ith_distances.sort()
    ith_distances = ith_distances[-tail_size:]

    aux_model = WeibullScipy()
    aux_model.fit(ith_distances)

    weibull_models['weibull_model'][weibull_models['y'] == i_class] = aux_model

# weibull_models['weibull_model'][0].predict(X = [0.01, 0.02, 0.1, .5])
# weibull_models['weibull_model'][1].predict(X = [0.01, 0.02, 0.1, .5])
# weibull_models['weibull_model'][0].pars_dict

weibull_models_dict = weibull_models.set_index(['y']).to_dict(orient='index')

weibull_models_dict
total_da

# -- CDFs on the distances -- #
for ith in weibull_models_dict.keys():
    total_da['cdf_' + ith] = weibull_models_dict[ith]['weibull_model'].predict(X=total_da['dist_to_mean_'+ith])

# -- alphas -- #

total_da[[i for i in total_da.columns if 'av_' in i]].apply(compute_alpha_weights, axis = 1)
compute_alpha_weights
for av in [i for i in total_da.columns if 'av_' in i]
# av_0, av_1

def wei(v):
    # v = [1, 2]
    len(v)
    # entrar no deles e ver como fizeram os pesos.

total_da[[i for i in total_da.columns if 'cdf_' in i]].apply(lambda x: len(x)-, axis=1)


# ------------------------------------------------ #
# ------------------------------------------------ #
# --
# ------------------------------------------------ #
# ------------------------------------------------ #

weibull_models