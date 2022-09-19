
import os
os.path
import sys
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/metrics/__init__.py')
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
from ostools import metrics as osm
from ostools.metrics import os_precision
from ostools.metrics import *
osm.os_precision
os_precision
get_os_conf_mat_terms


import pandas as pd
aux_df = pd.read_csv('/Users/leonardomarques/Downloads/aux_df.csv')
train_aux_df = aux_df[aux_df['split'] == 'train']
val_aux_df = aux_df[aux_df['split'] == 'val']
test_aux_df = aux_df[aux_df['split'] == 'test']

test_aux_df.head(5)
y_true_ = test_aux_df['group']
y_pred_ = test_aux_df['predictions']

get_metrics(
    y_true_
    , y_pred_
    , metric='precision'
    , average='macro'
)

get_metrics(
    y_true_
    , y_pred_
    , metric='recall'
    , average='macro'
)

get_metrics(
    y_true_
    , y_pred_
    , metric='true_negative_rate'
    , average='micro'
)


tnr = get_metrics(
    y_true_
    , y_pred_
    , metric='true_negative_rate'
    , average='micro'
)

recall =  get_metrics(
    y_true_
    , y_pred_
    , metric='recall'
    , average='micro'
)


recall =  get_metrics(
    y_true_
    , y_pred_
    , metric='recall'
    , average='micro'
)

youdens =  os_youdens_index(
    y_true_
    , y_pred_
    , average='micro'
)

get_metrics(
    y_true_
    , y_pred_
        , unknown_label=-1
        , average='macro'
        , metric='true_negative_rate')