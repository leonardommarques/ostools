import numpy as np
import pandas as pd
import sys


pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 1000)
pd.set_option("display.max_rows", 22)
pd.set_option("display.max_rows", 5)

sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/third_party_repository/')
import ostools
from ostools.models.OpenMax_Bendale import OpenMaxBendale

import scipy
# # ---------------------------------------------
om_bend = OpenMaxBendale(
    trained_model = model
    , tail_size=10
    , classes_to_revise=-1
    , distance_function=scipy.spatial.distance.euclidean
)
self = om_bend

om_bend.fit(
    X = total_da.query('split == "train"')[['x1', 'x2']]
    # , y = pd.get_dummies(total_da.query('split == "train"')[['y']])
    , y = total_da.query('split == "train"')['y']
)


labels_ = list(om_bend.labels) + [-1]
ombend_preds = om_bend.predict_proba(total_da[['x1', 'x2']])
ombend_preds['pred_class'] = om_bend.predict(total_da[['x1', 'x2']])
ombend_preds['y'] = total_da['y']
ombend_preds['split'] = total_da['split'].values

om_bend_quality_df = metrics.metrics_df(
    df = ombend_preds
    , observed_col = 'y'
    , predicted_col = 'pred_class'
    , split_col = 'split'
    , unknown_label = -1
)

om_bend_quality_df

