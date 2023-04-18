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
#from OSDN.utils.compute_openmax import computeOpenMaxProbability

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Softmax
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import plotnine as gg

from datetime import datetime
from scipy.spatial.distance import  euclidean
# ------------------------------------------------------------ #
# -- data
# ------------------------------------------------------------ #

data_folder = '/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my weibull EVM/example_data/sets/'
data_sets = os.listdir(data_folder)
data_sets = ['0-2-3-6']

finished_data_folder = '/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/experiments/EVM_10_classes/'
finished_data_sets = os.listdir(finished_data_folder)

i_data_set = -1
while i_data_set < len(data_sets)-1:
    i_data_set = i_data_set + 1

    data_set_name = data_sets[i_data_set]
    print(f'{i_data_set} / {len(data_sets)}: {round(i_data_set / len(data_sets), 3)}%   -   Dataset: {data_set_name}')

    # -- skip if dataset has already been analyzed -- #
    if data_set_name in finished_data_sets:
        continue

    # --------------------------------------- #
    # -- load data -- #
    # --------------------------------------- #
    train_da = pd.read_csv(data_folder + data_set_name + '/' + 'train.csv')
    test_da = pd.read_csv(data_folder + data_set_name + '/' + 'test.csv')

    feature_cols = ['x1', 'x2']
    target_col = 'group'

    X_train = train_da[feature_cols].values
    X_test = test_da[feature_cols].values

    y_train = train_da[target_col].values
    Y_train = pd.get_dummies(y_train)

    y_test = test_da[target_col].values
    Y_test = pd.get_dummies(y_test)

    data_splits = {
        'train': {'X': X_train, 'y': y_train}
        , 'test': {'X': X_test, 'y': y_test}
    }

    # ------------------------------------ #
    # -- data processing
    # ------------------------------------ #
    scaler = StandardScaler()
    scaler.fit(X_train)

    i_split = -1
    while i_split < len(data_splits) -1:
        i_split = i_split + 1

        data_splits_names = list(data_splits.keys())

        aux_X = data_splits[data_splits_names[i_split]]['X']
        aux_X = scaler.transform(aux_X)

        data_splits[data_splits_names[i_split]]['X'] = aux_X

    # ------------------------------------------------------------ #
    # -- model
    # ------------------------------------------------------------ #

    sk_evm = SKLEARN_EVM(
        tail_size=10
        # , cover_threshold=100 # set to 100 so that it uses all points to create weibull models (Extreme vectors)
        , cover_threshold=100  # set to 100 so that it uses all points to create weibull models (Extreme vectors)
        , distance_function=euclidean
        , distance_multiplier=1  # I still do not knwo why we should rescale the distances. Easier to compute??
        , confidence_threshold = 0.9
    )

    t0 = datetime.now()
    sk_evm.fit(X_train, y_train)
    tf = datetime.now()
    fit_time = (tf - t0).total_seconds()
    print(f'Time to Fit : {round(fit_time, 1)}s')

    # ------------------------------------------- #
    # -- evaluate
    # ------------------------------------------- #
    sk_model = sk_evm

    i_split = -1
    while i_split < len(data_splits) - 1:
        i_split = i_split + 1

        data_splits_names = list(data_splits.keys())
        data_split_name = data_splits_names[i_split]

        aux_X = data_splits[data_split_name]['X']
        aux_y = data_splits[data_split_name]['y']

        preds_df = sk_model.predict_proba(aux_X, return_df=True)
        preds_df = pd.DataFrame(preds_df)

        labels_ = np.unique(y_train)
        preds_df.columns = labels_
        preds_df.columns = ['proba_class_' + str(col_) for col_ in preds_df.columns]

        preds_df['pred_class'] = preds_df.apply(lambda xx: labels_[np.argmax(xx)], 1)

        data_splits[data_split_name]['pred'] = preds_df


    da_list = []
    for key_, value_ in data_splits.items():
        # key_ = list(data_splits.keys())[0]
        # value_ =list(data_splits.values())[0]

        aux_df = pd.DataFrame(value_['X'])
        aux_df.columns = ['x' + str(xx) for xx in aux_df.columns]

        aux_df['y'] = value_['y']
        aux_df[value_['pred'].columns] = value_['pred']
        aux_df['split'] = key_
        da_list.append(aux_df)

    total_da = pd.concat(da_list)

    evm_quality_df = metrics.metrics_df(
        df=total_da
        , observed_col='y'
        , predicted_col='pred_class'
        , split_col='split'
    )

    evm_quality_df

# total_da.to_csv(finished_data_folder + 'total_da_EVM.csv')
# total_da_old.to_csv(finished_data_folder + 'total_da_old.csv')


# del om_bend

# model.fit(
#     total_da.query('split == "train"')[['x0', 'x1']]
#     , y = pd.get_dummies(total_da.query('split == "train"')['y'])
#     , epochs=10
# )

from ostools.models.OpenMax_Bendale import OpenMaxBendale

om_bend = OpenMaxBendale(
    trained_model = model
    # trained_model = rn_model
    , tail_size=10
    , classes_to_revise=-1
    , distance_function=scipy.spatial.distance.euclidean
)


om_bend.fit(
    X = total_da.query('split == "train"')[['x0', 'x1']]
    , y = total_da.query('split == "train"')['y']
)


# om_bend.fit(
#     X = total_da_old.query('split == "train"')[['x1', 'x2']]
#     , y = total_da_old.query('split == "train"')['y'].astype(int)
# )

aux_preds = om_bend.predict(X = total_da[['x0', 'x1']])
total_da['pred_om_bend'] = aux_preds.values

total_da.to_csv(finished_data_folder + 'total_da_EVM.csv')
total_da_old.to_csv(finished_data_folder + 'total_da_old.csv')

quality_df_om = metrics.metrics_df(
        df=total_da
        , observed_col='y'
        , predicted_col='pred_om_bend'
        , split_col='split'
    )


quality_df

evm_quality_df
quality_df_om



from sklearn.metrics import confusion_matrix
confusion_matrix(
    total_da['y']
    , total_da['pred_om_bend'])


total_da['']

(
    gg.ggplot(
        total_da
        , gg.aes(x = 'x0'
                 , y = 'x1'
                 , fill = 'y'
                 )
    ) +
    gg.geom_point()
)



total_da['pred_class'].unique()
# total_da = pd.read_csv("/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my weibull EVM/example_data/toy_example_10_classes.txt")
# total_da['y'] = total_da['group']
#model_old = copy.deepcopy(model)

evm_quality_df['method'] = 'EVM'
quality_df_om['method'] = 'OpenMax'

evm_om_quality_df = pd.concat([quality_df_om, evm_quality_df])
evm_om_quality_df.to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/evm_om_quality_df.csv")

# ---------------------------------------------------- #
# -- Probability surfaces
# ---------------------------------------------------- #

x_surface = np.array(np.meshgrid(
    np.linspace(
        total_da['x0'].min()
        , total_da['x0'].max()
        , num=60
    )
    , np.linspace(
        total_da['x1'].min()
        , total_da['x1'].max()
        , num=60
    )
)).reshape(2, -1).T

x_surface = pd.DataFrame(x_surface)
x_surface.columns = ['x0', 'x1']
surface_preds_proba = om_bend.predict_proba(x_surface[['x0', 'x1']])
surface_preds_proba.columns = ['om_proba_' + col_ for col_ in surface_preds_proba.columns]
x_surface['max_probability'] = surface_preds_proba.apply(max, 1)
x_surface['prediction'] = om_bend.predict(x_surface[['x0', 'x1']])
x_surface[surface_preds_proba.columns] = surface_preds_proba

x_surface['prediction'].unique()
/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/misc/studies/EVM_10_classes.py
x_surface.to_csv("/Volumes/hd_Data/Users/leo/Documents/temp/x_surface.csv")

# --------------------------------------------------------------------------------- #
# -- 3d plot
# --------------------------------------------------------------------------------- #
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import pandas as pd
import numpy as np

# -- colors for the surfaces
surfacecolor_dic = {'-1': "#F8766D"
    , '0': "#A3A500"
    , '2': "#00BF7D"
    , '3': "#00B0F6"
    , '6': "#E76BF3"
                    }

surfaces = []
for prob_col in [col_ for col_ in x_surface.columns if 'om_proba_' in col_]:

    class_ = prob_col.replace('om_proba_', '')
    aux_df = x_surface[['x0', 'x1']].copy()
    aux_df['z'] = x_surface[[prob_col]]
    aux_df_wide = aux_df.pivot(index='x0', columns='x1', values='z')

    aux_surface = go.Surface(
        x=aux_df_wide.index
        , y=aux_df_wide.columns
        , z=aux_df_wide.values
        , surfacecolor=[int(class_)] * aux_df.shape[0]
        , colorscale=[surfacecolor_dic[class_], surfacecolor_dic[class_]]
    )

    surfaces.append(aux_surface)

    del aux_df, aux_df_wide


fig = go.Figure(data=surfaces)
fig.show()

