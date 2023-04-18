# ------------------------------------ #

import scipy
import numpy as np
import pandas as pd

import sys
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')

import ostools
from ostools.models import SKLEARN_EVM


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
class1_df['y'] = 'class_0'

class2_df = pd.DataFrame(class2)
class2_df['y'] = 'class_1'

class3_df = pd.DataFrame(class3)
class3_df['y'] = 'class_2'
df = pd.concat([class1_df, class2_df, class3_df])
df.columns = ['x0', 'x1', 'y']



df_test = pd.concat([pd.DataFrame(i) for i in [class1_test, class2_test, class3_test, class_other_test]])
df_test['y'] = np.concatenate([[0]*10, [1]*10, [2]*10, [-1]*10])
df_test.columns = df.columns

features = [i for i in df.columns if i not in ['y']]


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

# ----------------------------------------------- #
# -- Fit
# ----------------------------------------------- #
sk_evm = SKLEARN_EVM(
    tail_size=10
    # , cover_threshold=100 # set to 100 so that it uses all points to create weibull models (Extreme vectors)
    , cover_threshold=100 # set to 100 so that it uses all points to create weibull models (Extreme vectors)
    , distance_function=scipy.spatial.distance.euclidean
    , distance_multiplier=1  # I still do not knwo why we should rescale the distances. Easier to compute??
)

X = df[features].values
y = df['y'].values
sk_evm.fit(X, y)

# ------------------------------------ #
# -- predictions
# ------------------------------------ #
sk_evm.predict_proba(X, return_df=True)
sk_evm.predict(X, return_df=True)

X_test = df_test[features].values
sk_evm.predict_proba(X_test, return_df=True)
sk_evm.predict(X_test, return_df=True)


