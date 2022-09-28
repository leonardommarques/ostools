################################################################################

################################################################################
import pandas as pd
import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
from ostools.models import EVM
from datetime import datetime

# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
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


# -------------------------------------- #
# -- machine learning
# -------------------------------------- #
evm = EVM(
    margin_scale=1
    , tail_size=20
    , n_obs_to_fuse=4
    , confidence_threshold=0.9
)


t0 = datetime.now()
evm.fit(
    X_train
    , y_train
)
t1 = datetime.now()
time_fit = (t1-t0).total_seconds()
del t0

# ------------------------------ #
# -- predictions
# ------------------------------ #

predicted_labels = evm.predict(X_train)
train_da['prediction'] = predicted_labels


# -- validations
t0 = datetime.now()
predicted_labels = evm.predict(X_val)
val_da['prediction'] = predicted_labels
time_val = (datetime.now()-t0).total_seconds()
del t0

# -- test
t0 = datetime.now()
predicted_labels = evm.predict(X_test)
test_da['prediction'] = predicted_labels
time_test = (datetime.now()-t0).total_seconds()
del t0

# --------------------------------- #
# -- quality metrics
# --------------------------------- #

[os_accuracy(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]
[os_recall(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]
[os_precision(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]

