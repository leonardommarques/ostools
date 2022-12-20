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
np.random.seed(131)

class1 = np.random.normal((0,0),3,(50,2))
class2 = np.random.normal((-10,10),3,(50,2))
class3 = np.random.normal((10,-10),3,(50,2))

class1_test = np.random.normal((0,0),3,(10,2))
class2_test = np.random.normal((-10,10),3,(10,2))
class3_test = np.random.normal((10,-10),3,(10,2))
class_other_test = np.random.normal((-15,-10),3,(10,2))


class1_df = pd.DataFrame(class1)
class1_df['y'] = 0

class2_df = pd.DataFrame(class2)
class2_df['y'] = 1

class3_df = pd.DataFrame(class3)
class3_df['y'] = 2
# df = pd.concat([class1_df, class2_df, class3_df])
df = pd.concat([class1_df.head(10), class2_df.head(10), class3_df.head(10)])
df.columns = ['x0', 'x1', 'y']



df_test = pd.concat([pd.DataFrame(i) for i in [class1_test, class2_test, class3_test, class_other_test]])
df_test['y'] = np.concatenate([[0]*10, [1]*10, [2]*10, [-1]*10])
df_test.columns = df.columns

features = [i for i in df.columns if i not in ['y']]


# ------------- #
# -- split
# ------------- #
X_train = df[features].values
y_train = df['y'].values

X_test = df_test[features].values
y_test = df_test['y'].values


# ------------------------------------- #
# -- exploratory analysis
# ------------------------------------- #

fig = px.scatter(
    df
    , x="x0"
    , y="x1"
    , color="y"
    # , hover_data=['x0', 'x1', ]
)
# fig.show()

# ---------------------------------------------------------------------------------- #
# -- Model
# ---------------------------------------------------------------------------------- #
dir(models)

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
    predictions = pd.concat([df_test.reset_index(drop=True), predictions, predictions_classes, ], axis=1)

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


pd.DataFrame(quality_dict)

