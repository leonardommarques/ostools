################################################################################

################################################################################

import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import plotnine as gg

import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
# from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
# from ostools.models import EVM
# from datetime import datetime


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Softmax
import tensorflow as tf

from sklearn.model_selection import train_test_split

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
train_da, val_da = train_test_split(known_da, test_size=0.2, random_state=876357463)

X_train = train_da[['x1', 'x2']].values
X_val = val_da[['x1', 'x2']].values

y_train = train_da['y'].values
Y_train = pd.get_dummies(y_train)
y_val = val_da['y'].values
Y_val = pd.get_dummies(y_val)

test_da = pd.concat([test_da, db1])
X_test = test_da[['x1', 'x2']].values
y_test = test_da['y'].values
Y_test = pd.get_dummies(y_test)

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

model.fit(X_train, Y_train, epochs=20)
model.evaluate(X_train, Y_train)
model.evaluate(X_val, Y_val)
# model.evaluate(X_test, Y_test)

# ------------------------------------------- #
# -- Activations and predicitons
# ------------------------------------------- #
X_batch = X_train[:5].copy()

def get_activations(original_model, X_batch, layer = -2):
    from keras.models import Model

    intermediate_layer_model = Model(
        inputs=original_model.input
        , outputs=original_model.get_layer(index=layer).output)

    activations = intermediate_layer_model.predict(X_batch)

    return activations



av = get_activations(model, X_batch=X_batch)
np.apply_along_axis(lambda x: [np.exp(i)/sum(np.exp(x)) for i in x], 1, av)
model.predict(X_batch)

# ------------------------------------------- #
# -- Predictions and Activation Vectors
# ------------------------------------------- #

preds = model.predict(total_da[['x1', 'x2']].values)
avs = get_activations(model, X_batch=total_da[['x1', 'x2']].values)
for i in range(preds.shape[1]-1):
    # - predicitons
    total_da['proba_' + str(i)] = preds[:, i]
    # -- AVs
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
    , gg.aes(x = 'av_0', y = 'av_1', fill = 'y')
        ) +
        gg.geom_point(size = 4)
)
