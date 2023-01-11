import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 1000)
pd.set_option("display.max_rows", 22)
pd.set_option("display.max_rows", 5)

# # ---------------------------------------------
om_bend = OpenMax_Bendale(
    trained_model = model
    , tail_size=10
    , classes_to_revise=-1
    , distance_function=scipy.spatial.distance.euclidean
)

om_bend.fit(
    X = total_da.query('split == "train"')[['x1', 'x2']]
    , y = pd.get_dummies(total_da.query('split == "train"')[['y']])
)

labels_ = list(om_bend.labels) + [-1]
ombend_preds = om_bend.predict_proba(total_da[['x1', 'x2']])
ombend_preds['pred'] = ombend_preds.apply(lambda xx: labels_[np.argmax(xx)], 1)
ombend_preds['pred_class'] = om_bend.predict(total_da[['x1', 'x2']])
ombend_preds['y'] = total_da['y'].astype(int).values
ombend_preds['split'] = total_da['split'].values

om_bend_quality_df = metrics.metrics_df(
    df = ombend_preds
    , observed_col = 'y'
    , predicted_col = 'pred'
    , split_col = 'split'
)

om_bend_quality_df