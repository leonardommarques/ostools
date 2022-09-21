

import os
os.path
import sys
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/metrics/__init__.py')
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/ostools/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
from ostools import metrics as osm
from ostools.metrics import os_precision
from ostools.metrics import *
from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index

osm.os_precision
os_precision
dir(osm)


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
import pandas as pd
# sample_train = pd.read_csv('/Users/leonardomarques/Downloads/sample_train.csv')
# sample_train = pd.read_csv('/Users/leonardomarques/Downloads/sample_train.csv')
# X = sample_train[['x1', 'x2']].values
# y = sample_train['group'].values


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
predicted_labels = predicted_labels.reshape(-1, 1)
train_da['prediction'] = predicted_labels

from sklearn.metrics import confusion_matrix
confusion_matrix(train_da['group'], train_da['prediction'], labels = np.sort(train_da['group'].unique()))
train_da['group'].value_counts()

sum(train_da['group'].value_counts())
332513078856332513078856332513078856

probas_ = evm.predict_proba(X_val)
predicted_labels = evm.predict(X_val)
predicted_labels = predicted_labels.reshape(-1, 1)
val_da['prediction'] = predicted_labels
confusion_matrix(val_da['group'], val_da['prediction'], labels = np.sort(val_da['group'].unique()))


probas_ = evm.predict_proba(X_test)
predicted_labels = evm.predict(X_test)
predicted_labels = predicted_labels.reshape(-1, 1)
test_da['prediction'] = predicted_labels



test_da['prediction'].unique()
test_da['prediction'].value_counts()

train_da['prediction'].value_counts()
val_da['prediction'].value_counts()

self = evm
return_dict = True
from datetime import datetime

t0 = datetime.now()
probas_ = evm.predict_proba(X_train)
t_pred = datetime.now()

tempos_pred = [t0, t_pred]

t0 = datetime.now()
probas_ = evm.predict(X_train)
t_pred2 = datetime.now()
tempos_pred2 = [t0, t_pred2]

# -------------------------------------------------- #
# -------------------------------------------------- #
# debug fit
# -------------------------------------------------- #
# -------------------------------------------------- #
X = X_train
y = y_train
self = evm
# evm.fit(
#     X_train
#     , y_train
# )
len(self.weibulls_dict)
self.weibulls_dict[1]

weibulls_dict[1]

self.weibulls_dict[1]
sum([len(i) for i in self.weibulls_dict.values()])
self.weibulls_dict.keys()
tempos = [
    t0
    , t_filtros
    , t_tail
    , t_fit
]
datetime.

(tempos[1]-tempos[0]).total_seconds()
(tempos[2]-tempos[1]).total_seconds()
(tempos[3]-tempos[2]).total_seconds()


t0 = datetime.now()
stats.exponweib.fit(obs_distances, floc=0, f0=1)
tfit2 = datetime.now()
(tfit2-t0).total_seconds()
np.array([obs_distances, obs_distances, obs_distances])

t0 = datetime.now()
stats.exponweib.fit(obs_distances, floc=0, f0=1, loc = 1, scale = 1)
tfit2 = datetime.now()
(tfit2-t0).total_seconds()



y_true = test_da['group']
y_pred = test_da['predictions']

os_accuracy(y_true, y_pred)


y_true = train_da['group']
y_pred = train_da['predictions']

os_accuracy(y_true, y_pred)