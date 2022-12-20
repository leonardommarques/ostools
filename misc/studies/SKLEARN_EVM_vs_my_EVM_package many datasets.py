# ------------------------------------ #

import scipy
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 1000)
pd.set_option("display.max_rows", 5)


import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')

import ostools
from ostools import models
from ostools import metrics
# from ostools.models import SKLEARN_EVM, EVM


import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"



# ------------------------------------- #
# -- Data
# ------------------------------------- #

data_path = "/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my weibull EVM/example_data/toy_example_10_classes.txt"
original_df = pd.read_csv(data_path)
original_df.columns = ['y', 'x1', 'x2']

# ------------------------------- #
# -- split train test
# ------------------------------- #

unknown_labels = [8, 9]
unknown_df = original_df[original_df['y'].isin(unknown_labels)]
unknown_df['y'] = -1
known_df = original_df[~ original_df['y'].isin(unknown_labels)]

np.random.seed(131)
train_idx = np.random.choice(
    [True, False]
    , known_df.shape[0]
)


train_df = known_df[train_idx]
test_df = known_df[~train_idx]
test_df = pd.concat([test_df, unknown_df])

features = [i for i in train_df.columns if i not in ['y']]


# ------------- #
# -- split
# ------------- #
X_train = train_df[features].values
y_train = train_df['y'].values

X_test = test_df[features].values
y_test = test_df['y'].values

# ------------------------------------- #
# -- exploratory analysis
# ------------------------------------- #

fig = px.scatter(
    train_df
    , x="x1"
    , y="x2"
    , color="y"
    # , hover_data=['x0', 'x1', ]
)
# fig.show()

# ---------------------------------------------------------------------------------- #
# -- Model
# ---------------------------------------------------------------------------------- #

# -- My EVM implementation -- #
my_evm = models.EVM(
    margin_scale=1
    , tail_size=10
    , n_obs_to_fuse=4
    , confidence_threshold=0.9
)

# -- My version of pip Multi EVM -- #

sk_evm = models.SKLEARN_EVM(
    margin_scale=1
    , tail_size=10
    , n_obs_to_fuse=4
    , confidence_threshold=0.9
)

MODELS = {
    'my_evm': my_evm
    , 'sk_evm': sk_evm
}

# sk_evm.fit()
# my_evm.fit(X_train, y_train)
type(sk_evm)

for key_, value_ in MODELS.items():
    print('Fitting ' + key_)
    value_.fit(X_train, y_train)

del key_, value_

# -------------------------------- #
# -- predictions
# -------------------------------- #
predictions_dict = {}
for key_, value_ in MODELS.items():
    # break
    print('Predicting ' + key_)

    # X_pred = X_train
    X_pred = X_test
    predictions = value_.predict_proba(X_pred, return_df = True)
    predictions.columns = [f"pred_class_{i}" for i in predictions.columns]

    predictions_classes = value_.predict(X_pred, return_df = True)
    predictions_classes.columns = ['prediction']

    # predictions = pd.concat([df.reset_index(drop=True), predictions, predictions_classes,], axis=1)
    predictions = pd.concat([test_df.reset_index(drop=True), predictions, predictions_classes, ], axis=1)

    predictions['model'] = key_

    predictions_dict[key_] = predictions


# ------------------------------------------- #
# -- Metrics
# ------------------------------------------- #


# dir(metrics)
# 'np',
#  'os_accuracy',
#  'os_precision',
#  'os_recall',
#  'os_true_negative_rate',
#  'os_youdens_index'

quality_dict = {}
for key_, value_ in predictions_dict.items():

    quality_dict[key_] = {}

    # -- os_true_negative_rate
    os_true_negative_rate_ = metrics.os_true_negative_rate(
        predictions_dict[key_]['y']
        , predictions_dict[key_]['prediction']
        , unknown_label=-1
        , labels=None
        , average='macro'
    )

    # -- precision
    os_precision_ = metrics.os_precision(
        predictions_dict[key_]['y']
        , predictions_dict[key_]['prediction']
        , unknown_label=-1
        , labels=None
        , average='macro'
    )

    # -- os Recall
    os_recall_ = metrics.os_recall(
        predictions_dict[key_]['y']
        , predictions_dict[key_]['prediction']
        , unknown_label=-1
        , labels=None
        , average='macro'
    )

    # -- os accuracy
    os_accuracy_ = metrics.os_accuracy(
        predictions_dict[key_]['y']
        , predictions_dict[key_]['prediction']
        , unknown_label=-1
        # , labels=None
    )

    quality_dict[key_]['os_precision'] = os_precision_
    quality_dict[key_]['os_recall'] = os_recall_
    quality_dict[key_]['os_accuracy'] = os_accuracy_
    quality_dict[key_]['true_negative_rate'] = os_true_negative_rate_


quality_dict

pd.DataFrame(quality_dict)

