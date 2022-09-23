pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
from ostools import metrics as osm
from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
from ostools.models import EVM
from sklearn.metrics import confusion_matrix


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

################################################################################
################################################################################
################################################################################
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
from ostools import metrics as osm
from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
from ostools.models import EVM


train_da = pd.read_csv('/Users/leonardomarques/Downloads/train_da.csv')
X_train = train_da[['x1', 'x2']].values
y_train = train_da['group'].values

val_da = pd.read_csv('/Users/leonardomarques/Downloads/val_da.csv')
X_val = val_da[['x1', 'x2']].values
y_val = val_da['group'].values

test_da = pd.read_csv('/Users/leonardomarques/Downloads/test_da.csv')
X_test = test_da[['x1', 'x2']].values
y_test = test_da['group'].values

train_da['group'].value_counts()

evm = EVM(
    margin_scale=1
    , tail_size=20
    , n_obs_to_fuse=4
    , confidence_threshold=0.9
)

from datetime import datetime
t0 = datetime.now()
evm.fit(
    X_train
    , y_train
)
t1 = datetime.now()
tempo_fit = (t1-t0).total_seconds()

tempo_fit/60
# -- check some weibull fits
evm.weibulls_dict[1][0]

probas_ = evm.predict_proba(X_train)
predicted_labels = evm.predict(X_train)
train_da['prediction'] = predicted_labels
# confusion_matrix(train_da['group'], train_da['prediction'], labels = np.sort(train_da['group'].unique()))


predicted_labels = evm.predict(X_val)
predicted_labels = predicted_labels.reshape(-1, 1)
val_da['prediction'] = predicted_labels
# confusion_matrix(val_da['group'], val_da['prediction'], labels = np.sort(val_da['group'].unique()))

predicted_labels = evm.predict(X_test)
test_da['prediction'] = predicted_labels
# confusion_matrix(test_da['group'], test_da['prediction'], labels = np.sort(test_da['group'].unique()))


test_da['predictions'].value_counts()
val_da['predictions'].value_counts()
test_da['predictions'].value_counts()
# --------------------------------- #
# -- metrics
# --------------------------------- #
from ostools.metrics  import os_recall, os_precision

[os_accuracy(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]
[os_recall(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]
[os_precision(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]

os_recall(train_da['group'], train_da['predictions'])
y_true = train_da['group']
y_pred = train_da['predictions']
metric = 'recall'



[os_accuracy(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]
[get_metrics(y_true = a['group'], y_pred = a['predictions'], unknown_label = -1, labels=None, average = 'macro', metric = 'recall')  for a in [train_da, val_da, test_da]]
[get_metrics(y_true = a['group'], y_pred = a['predictions'], unknown_label = -1, labels=None, average = 'macro', metric = 'precision')  for a in [train_da, val_da, test_da]]

[os_recall(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]


os_accuracy(test_da['group'], test_da['predictions'])