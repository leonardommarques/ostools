################################################################################
# EVM vs OpenMax
# compare the both models
################################################################################
import copy
import os
import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import sys

sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/third_party_repository/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/third_party_repository/OSDN')

from ostools.models import SKLEARN_EVM
from ostools import metrics
from ostools.models.OpenMax_Bendale import OpenMaxBendale
# from OSDN.utils.compute_openmax import computeOpenMaxProbability

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Softmax
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import numpy as np
import plotnine as gg
import scipy

from datetime import datetime
from scipy.spatial.distance import euclidean

# ------------------------------------------------------------ #
# -- data
# ------------------------------------------------------------ #
# sample size

n = 100

populations_pars = {
    '0': {
        'mean': [1, 1]
        , 'sd': [1, 1]
    }
    , '1': {
        'mean': [5, 5]
        , 'sd': [1, 1]
    }

    , '2': {
        'mean': [15, 15]
        , 'sd': [1, 1]
    }
}

datasets = {
    key_: pd.DataFrame({
        'x0': np.random.normal(value_['mean'][0], value_['sd'][0], n)
        , 'x1': np.random.normal(value_['mean'][1], value_['sd'][1], n)
        , 'y': [key_] * n
    })
    for key_, value_ in populations_pars.items()
}
data = pd.concat(datasets)

(
        gg.ggplot(
            data
            , gg.aes(x='x0', y='x1', fill='y')
        ) +
        gg.geom_point()
)

# ------------------------------------------------ #
# -- splits
# ------------------------------------------------ #

all_labels = list(data['y'].unique())
known_labels = all_labels[:2]
data['y'][~data['y'].isin(known_labels)] = -1

[[80 * .8, 80 * .2], 20]
idx_train_test_val = np.random.choice(['train', 'val', 'test'], data.shape[0], p=[.64, .16, .2])
data['split'] = idx_train_test_val
data['split'][~data['y'].isin(known_labels)] = 'test'

data.groupby(['split', 'y']).size()

features = ['x0', 'x1']

X_train = data[data['split'] == 'train'][features].values
X_val = data[data['split'] == 'val'][features].values
X_test = data[data['split'] == 'test'][features].values

y_train = data[data['split'] == 'train']['y'].values
y_val = data[data['split'] == 'val']['y'].values
y_test = data[data['split'] == 'test']['y'].values

Y_train = pd.get_dummies(y_train)
Y_val = pd.get_dummies(y_val)
# y_val
# y_test

# ------------------------------------------------------------ #
# -- model
# ------------------------------------------------------------ #

model = Sequential()

model.add(Dense(
    input_shape=(X_train.shape[1],)
    , units=10
    , activation='sigmoid'
))

model.add(Dense(
    units=5
    , activation='sigmoid'
))

# ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - ! - #
# !!!!!!!!!!!!! It is imperative that the two last layers are a linear and a softmax. So that we can get the
# activation vectors of the penultimate layer

model.add(Dense(
    units=Y_train.shape[1]
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
    optimizer=keras.optimizers.Adam(learning_rate=0.1)
    , loss=keras.losses.CategoricalCrossentropy()
    , metrics=['accuracy']
)

dir(model)
model.fit(X_train, Y_train, epochs=10)

model.evaluate(X_train, Y_train)
model.evaluate(X_val, Y_val)

# ------------------------------- #
# -- OpenMax
# ------------------------------- #

om_bend = OpenMaxBendale(
    trained_model=model
    , tail_size=10
    , classes_to_revise=-1
    , distance_function=scipy.spatial.distance.euclidean
)

om_bend.fit(
    X=X_train
    # , y = pd.get_dummies(total_da.query('split == "train"')[['y']])
    , y=y_train
)

probas = om_bend.predict_proba(data[features].values)
probas.columns = [f"proba_{i}" for i in probas.columns]
probas.values
data[probas.columns] = probas.values
data['predicted'] = om_bend.predict(data[features].values)

classes = known_labels + ['-1']
data['predicted_min'] = probas.apply(lambda xx: classes[np.argmin(xx)], 1).values

om_cdf = om_bend.predict_proba(data[features].values, return_cdf = True)
om_cdf = 1 - om_cdf
om_cdf.columns = [f"proba_{i}" for i in om_cdf.columns]
data[om_cdf.columns] = om_cdf.values

cdf_pred = om_cdf.apply(np.argmax, 1)
cdf_pred = cdf_pred.astype(str)

cdf_max = om_cdf.apply(max, 1)
data['cdf_max'] = cdf_max.values

aux_confidence_threshold = 0.7
cdf_pred[cdf_max >= aux_confidence_threshold] = -1
data['cdf_predicted'] = cdf_pred.values
# data[list(om_cdf.columns) + ['y']]

# -------------------------------------- #
# -- EVM
# -------------------------------------- #
sk_evm = SKLEARN_EVM(
    tail_size=10
    # , cover_threshold=100 # set to 100 so that it uses all points to create weibull models (Extreme vectors)
    , cover_threshold=100  # set to 100 so that it uses all points to create weibull models (Extreme vectors)
    , distance_function=euclidean
    , distance_multiplier=1  # I still do not knwo why we should rescale the distances. Easier to compute??
    , confidence_threshold=0.9
)

t0 = datetime.now()
sk_evm.fit(X_train, y_train)
tf = datetime.now()
fit_time = (tf - t0).total_seconds()
print(f'Time to Fit : {round(fit_time, 1)}s')

sk_evm.confidence_threshold
evm_probas = sk_evm.predict_proba(data[features].values)
evm_probas = pd.DataFrame(evm_probas)
evm_probas.columns = [f"evm_proba_{col}" for col in evm_probas.columns]
data[evm_probas.columns] = evm_probas.values
evm_predictions = sk_evm.predict(data[features].values).values
data['evm_predicted'] = evm_predictions
data['evm_predicted'] = data['evm_predicted'].astype(str)

np.unique(data['evm_predicted'], return_counts=True)
np.unique(data['evm_predicted'] == data['y'], return_counts=True)


confusion_matrix(
    data['y'].astype(int)
    , data['evm_predicted'].astype(int))
# -------------------------------------- #
# -- quality
# -------------------------------------- #
data['predicted'] = data['predicted'].astype(str)
data['y'] = data['y'].astype(str)

om_quality_df = metrics.metrics_df(
    df=data
    , observed_col='y'
    , predicted_col='predicted'
    , split_col='split'
)

evm_quality_df = metrics.metrics_df(
    df=data
    , observed_col='y'
    , predicted_col='evm_predicted'
    , split_col='split'
)

data['cdf_predicted'] = data['cdf_predicted'].astype(str)
cdf_quality_df = metrics.metrics_df(
    df=data
    , observed_col='y'
    , predicted_col='cdf_predicted'
    , split_col='split'
)

data['cdf_predicted'].unique()
evm_quality_df
om_quality_df
cdf_quality_df

# ------------------------------------------ #
# -- Surface analysis
# ------------------------------------------ #
x_surface = np.array(np.meshgrid(
    np.linspace(
        data['x0'].min()
        , data['x0'].max()
        , num=60
    )
    , np.linspace(
        data['x1'].min()
        , data['x1'].max()
        , num=60
    )
)).reshape(2, -1).T

# grid_range = 50
# x_surface = np.array(np.meshgrid(
#     np.linspace(
#         -grid_range
#         , grid_range
#         , num=100
#     )
#     , np.linspace(
#         -grid_range
#         , grid_range
#         , num=100
#     )
# )).reshape(2, -1).T

# # -- Diagonal -- # #
x_surface = np.zeros([10, 2])
x_surface[:, 0] = np.linspace(
        data['x0'].min()
        , data['x0'].max()
        , num=10
    )

x_surface[:, 1] = np.linspace(
        data['x1'].min()
        , data['x1'].max()
        , num=10
    )
# x_surface

x_surface = pd.DataFrame(x_surface)
x_surface.columns = ['x0', 'x1']
surface_preds_proba = om_bend.predict_proba(x_surface[['x0', 'x1']].values)
surface_preds_proba.columns = ['om_proba_' + col_ for col_ in surface_preds_proba.columns]
x_surface['max_probability'] = surface_preds_proba.apply(max, 1)
x_surface['predicted'] = om_bend.predict(x_surface[['x0', 'x1']])
x_surface[surface_preds_proba.columns] = surface_preds_proba

nn_surface_preds_proba = model.predict(x_surface[['x0', 'x1']].values)
nn_surface_preds_proba = pd.DataFrame(nn_surface_preds_proba)
nn_surface_preds_proba.columns = [f'nn_proba_{col_}' for col_ in nn_surface_preds_proba.columns]
x_surface['nn_max_probability'] = nn_surface_preds_proba.apply(max, 1)
x_surface['nn_predicted'] = nn_surface_preds_proba.apply(np.argmax, 1).astype(str)

evm_surface_preds_proba = sk_evm.predict_proba(x_surface[['x0', 'x1']].values)
evm_surface_preds_proba = pd.DataFrame(evm_surface_preds_proba)
evm_surface_preds_proba.columns = [f'evm_proba_{col_}' for col_ in evm_surface_preds_proba.columns]
x_surface['evm_max_probability'] = evm_surface_preds_proba.apply(max, 1)
x_surface['evm_predicted'] = sk_evm.predict(x_surface[['x0', 'x1']].values)
x_surface['evm_predicted'].unique()

cdf_surface_preds_proba = om_bend.predict_proba(x_surface[['x0', 'x1']].values, return_cdf = True)
cdf_surface_preds_proba = 1 - cdf_surface_preds_proba
cdf_surface_preds_proba = pd.DataFrame(cdf_surface_preds_proba)
cdf_surface_preds_proba.columns = [f'cdf_proba_{col_}' for col_ in cdf_surface_preds_proba.columns]
x_surface['cdf_max_probability'] = cdf_surface_preds_proba.apply(max, 1)
x_surface['cdf_predicted'] = cdf_surface_preds_proba.apply(np.argmax, 1)
x_surface['cdf_predicted'][x_surface['cdf_max_probability'] < aux_confidence_threshold] = '-1'
x_surface['cdf_predicted'] = x_surface['cdf_predicted'].astype(str)
x_surface['cdf_predicted'].unique()

# -- revised activations -- #

revised_surface_preds_proba = om_bend.predict_proba(x_surface[['x0', 'x1']].values, return_revised = True)
revised_surface_preds_proba = pd.DataFrame(revised_surface_preds_proba)
revised_surface_preds_proba.columns = [f'revised_av_{col_}' for col_ in revised_surface_preds_proba.columns]
x_surface[revised_surface_preds_proba.columns] = revised_surface_preds_proba


x_surface.to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/x_surface.csv")
data.to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/data.csv")
data['y'].unique()
