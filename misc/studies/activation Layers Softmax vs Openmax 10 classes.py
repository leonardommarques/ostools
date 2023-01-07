################################################################################
# Softmax VS OpenMax
# I try to implement OpenMax
################################################################################

### Hyper parameters
tail_size = 10

import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import plotnine as gg

import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/third_party_repository/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/third_party_repository/OSDN')
from OSDN.utils.compute_openmax import computeOpenMaxProbability

# from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
# from ostools.models import EVM
# from datetime import datetime
from ostools.functions import WeibullScipy
from ostools.functions import get_activations
from ostools.functions import compute_alpha_weights

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

np.random.seed(2956)
total_da = pd.read_csv("/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my weibull EVM/example_data/toy_example_10_classes.txt")
total_da['y'] = total_da['group']
total_da = total_da.drop(columns = ['group'])

# -- select known classes and relabel them -- #
known_classes = [2, 0, 3, 6]
known_classes = np.array(known_classes)
known_classes.sort()

total_da['y'][~total_da['y'].isin(known_classes)] = -1

# relabel classes according to order
known_classes_dict = {known_classes[j]: j for j in range(len(known_classes))}
known_classes_dict[-1] = -1

total_da['y'] = total_da['y'].apply(lambda xx: known_classes_dict[xx])

total_da['y'] = total_da['y'].apply(str)
known_da = total_da[total_da['y'] != '-1']

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
test_da = pd.concat([test_da, total_da.query('y == "-1"')])

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
    units = Y_train.shape[1]
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


model.fit(X_train, Y_train, epochs=10)
# model.fit(X_train, Y_train, epochs=1)
# model.fit(X_train, Y_train, epochs=20)
model.evaluate(X_train, Y_train)
model.evaluate(X_val, Y_val)

# ------------------------------------------- #
# -- Activations and predicitons
# ------------------------------------------- #
X_batch = X_train[:5].copy()

# activation vectors
av = get_activations(model, x_batch=X_batch)
# softmax on the activation vetors
np.apply_along_axis(lambda x: [np.exp(i)/sum(np.exp(x)) for i in x], 1, av)

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
#     , gg.aes(x ='av_0'
#              , y='av_1'
#              , fill='y')
#         ) +
#         gg.geom_point(size = 4) +
#         gg.facet_wrap('~split')
# )

# ------------------------------------------- #
# ------------------------------------------- #
# -- Mean activation vectors
# ------------------------------------------- #
# ------------------------------------------- #

# -- get mean activation vectors
## PAPER page 5: The model µi is computed using the images associated with category i, images that were classified correctly (top-1) during training process.
correctly_predicted_da = total_da[total_da['predicted_class'] == total_da['y']]
mavs_da = correctly_predicted_da.query('split == "train"')[['av_0', 'av_1', 'y']].groupby('y').mean().reset_index()

# -------------------------------------- #
# -- distance to MAV
# -------------------------------------- #
# import libmr

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

# -- alphas weights -- #
aux_alphas = total_da[[i for i in total_da.columns if 'av_' in i]].apply(compute_alpha_weights, axis = 1)
aux_alphas = np.concatenate(aux_alphas)
aux_alphas = pd.DataFrame(aux_alphas)
for col in aux_alphas.columns:
    total_da['alpha_weight_' + str(col)] = aux_alphas[col]

# ------------------------------------------------ #
# ------------------------------------------------ #
# -- Modified scores
# on this step they apply the weigths and modifications to the scores.
# ------------------------------------------------ #
# ------------------------------------------------ #

# -- modified scores, that are the activation vectors

classes = total_da['y'].unique()
classes.sort()
classes = list(classes[classes != '-1'])

for class_ in classes:
    # class_ = classes[0]
    aux_modified_fc8_score = total_da[f'av_{class_}'] * (1 - total_da[f'alpha_weight_{class_}']*total_da[f'cdf_{class_}'])
    total_da[f'revised_av_{class_}'] = aux_modified_fc8_score

# -- add revised activation for class "unknown" line 6 algorith 2
# openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score]
aux_revised_unknown = pd.DataFrame()
for class_ in classes:
    # class_ = classes[0]
    aux_revised_unknown[f'aux_revised_unknown_{class_}'] = total_da[f'av_{class_}'] - total_da[f'revised_av_{class_}']

revised_cols = [i for i in total_da.columns if 'revised_av' in i]
aux_revised_unknown

calculated_openMax = []
i = -1
while i < total_da.shape[0]-1:
    i = i+1
    aux_calculated_openMax = computeOpenMaxProbability(
        np.array(list([total_da[[i for i in total_da.columns if 'revised_av' in i]].values[i]]))
        , np.array(list([aux_revised_unknown.values[i]]))
    )

    calculated_openMax.append(aux_calculated_openMax)

calculated_openMax_df = pd.DataFrame(np.array(calculated_openMax))

i = -1
while i < len(calculated_openMax_df.columns)-1:
    i = i+1

    if i == len(calculated_openMax_df.columns)-1:
        total_da[f'openMax_prob_class_-1'] =  calculated_openMax_df.iloc[:,i]
    else:
        total_da[f'openMax_prob_class_{i}'] = calculated_openMax_df.iloc[:,i]


# -- add prediction -- #
openmax_cols = [i for i in total_da.columns if 'openMax_prob_class_' in i]
total_da['openMax_pred_class'] = total_da[openmax_cols].apply(lambda xx: openmax_cols[np.argmax(xx)].replace('openMax_prob_class_', ''), axis = 1)


# (
#         gg.ggplot(
#     total_da
#     , gg.aes(
#                 # x = 'openMax_prob_class_0', y = 'openMax_prob_class_1' #, 'openMax_prob_class_-1'
#                 x = 'av_0', y = 'av_1'
#                 # x = 'modified_av_0', y = 'modified_av_1'
#                 , fill = 'y'
#             )
#         ) +
#         gg.geom_point(size = 4)
# )

weibull_models['pars'] = weibull_models['weibull_model'].apply(lambda xx: xx.pars_dict)
weibull_models['pars'][0]

weibull_models['ppf_90'] = -10
i = -1
while i < weibull_models.shape[0]-1:
    i = i+1
    weibull_models['ppf_90'][i] = scipy.stats.weibull_min.ppf(
        .99
        , c=weibull_models['pars'][i]['shape']
        , loc=weibull_models['pars'][i]['loc']
        , scale=weibull_models['pars'][i]['scale']
                            )

weibull_models.columns
weibull_models.drop(columns=['weibull_model']).to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/weibull_models.csv")
total_da.to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/total_da.csv")

total_da['openMax_pred_class'].value_counts()

# ---------------------------------- #
# -- metrics -- #
# ---------------------------------- #
from ostools.metrics import get_metrics
from ostools import metrics

quality_df = metrics.metrics_df(
    total_da
    , observed_col = 'y'
    , predicted_col = 'openMax_pred_class'
    , split_col = 'split'
)

quality_df
quality_df.to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/quality_df.csv")

"""
CEMED8810297605

"""