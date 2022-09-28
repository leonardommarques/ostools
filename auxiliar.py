pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# import os
# import sys
# sys.path.append('/Volumes/hd_Data/Users/leo/Documents/Estudos/UTFPR/Orientacao/my_packages/')
# from ostools import metrics as osm
# from ostools.metrics import os_precision, os_recall, os_true_negative_rate, os_youdens_index, os_accuracy
# from ostools.models import EVM
# from sklearn.metrics import confusion_matrix
#
#
# import pandas as pd
# aux_df = pd.read_csv('/Users/leonardomarques/Downloads/aux_df.csv')
# train_aux_df = aux_df[aux_df['split'] == 'train']
# val_aux_df = aux_df[aux_df['split'] == 'val']
# test_aux_df = aux_df[aux_df['split'] == 'test']
#
# test_aux_df.head(5)
# y_true_ = test_aux_df['group']
# y_pred_ = test_aux_df['predictions']
#
# get_metrics(
#     y_true_
#     , y_pred_
#     , metric='precision'
#     , average='macro'
# )
#
# get_metrics(
#     y_true_
#     , y_pred_
#     , metric='recall'
#     , average='macro'
# )
#
# get_metrics(
#     y_true_
#     , y_pred_
#     , metric='true_negative_rate'
#     , average='micro'
# )
#
#
# tnr = get_metrics(
#     y_true_
#     , y_pred_
#     , metric='true_negative_rate'
#     , average='micro'
# )
#
# recall =  get_metrics(
#     y_true_
#     , y_pred_
#     , metric='recall'
#     , average='micro'
# )
#
#
# recall =  get_metrics(
#     y_true_
#     , y_pred_
#     , metric='recall'
#     , average='micro'
# )
#
# youdens =  os_youdens_index(
#     y_true_
#     , y_pred_
#     , average='micro'
# )
#
# get_metrics(
#     y_true_
#     , y_pred_
#         , unknown_label=-1
#         , average='macro'
#         , metric='true_negative_rate')

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
t0 = datetime.now()
# probas_ = evm.predict_proba(X_train)
predicted_labels = evm.predict(X_train)
train_da['prediction'] = predicted_labels
# confusion_matrix(train_da['group'], train_da['prediction'], labels = np.sort(train_da['group'].unique()))
tempo_train = (datetime.now()-t0).total_seconds()
del t0

t0 = datetime.now()
predicted_labels = evm.predict(X_val)
val_da['prediction'] = predicted_labels
# confusion_matrix(val_da['group'], val_da['prediction'], labels = np.sort(val_da['group'].unique()))
tempo_val = (datetime.now()-t0).total_seconds()
del t0


t0 = datetime.now()
predicted_labels = evm.predict(X_test)
test_da['prediction'] = predicted_labels
# confusion_matrix(test_da['group'], test_da['prediction'], labels = np.sort(test_da['group'].unique()))
tempo_test = (datetime.now()-t0).total_seconds()

print(f"""
tempo_train: {tempo_train}
tempo_val: {tempo_val}
tempo_test: {tempo_test}
""")

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


[os_recall(a['group'], a['predictions']) for a in [train_da, val_da, test_da]]


os_accuracy(test_da['group'], test_da['predictions'])

# -------------------------------------------------------- #
# -------------------------------------------------------- #
# -- brinks outro predict
# -------------------------------------------------------- #
# -------------------------------------------------------- #
t0 = datetime.now()
predicted_probas = evm.predict_proba(X_train)
# pd.DataFrame(predicted_probas)
predicted_probas = pd.DataFrame(predicted_probas)
# predictions = predicted_probas
predicted_labels = evm.predict(X_train)
train_da['prediction'] = predicted_labels
# confusion_matrix(train_da['group'], train_da['prediction'], labels = np.sort(train_da['group'].unique()))
tempo_train = (datetime.now()-t0).total_seconds()
del t0



predicted_labels = evm.predict_proba(X_train[:10, :], return_dict = True)
type(predicted_labels)
len(predicted_labels)
predicted_labels = evm.predict_proba(X_train[:5, :], return_dict = False)
train_da['prediction'] = predicted_labels
# confusion_matrix(train_da['group'], train_da['prediction'], labels = np.sort(train_da['group'].unique()))
tempo_train = (datetime.now()-t0).total_seconds()
del t0



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# debug this shit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# for class_key, class_value in weibulls_dict.items():
#     break

from scipy.spatial import distance_matrix
import copy
self = copy.deepcopy(evm)
X = X_train.copy()
# X = X + 0.01
X = X[:10]
return_dict = False
n_obs_to_fuse = 4
# predicted_labels = evm.predict(X_train)
weibulls_dict = copy.deepcopy(self.weibulls_dict)


labels = list(weibulls_dict.keys())
weibull_dfs = []
for class_key, class_value in weibulls_dict.items():
    # print(class_key)
    class_weibulls = weibulls_dict[class_key]
    for class_weibull in class_weibulls:
        # class_weibull = class_weibulls[0]
        class_weibull['weibull_pars']['params'] = str(class_weibull['weibull_pars']['params'] )
        class_weibull['weibull_pars'] = {key_:[value_] for key_, value_ in class_weibull['weibull_pars'].items()}
        class_weibull['weibull_pars'] = pd.DataFrame(class_weibull['weibull_pars'])
        class_weibull['label'] = pd.DataFrame({'label': [class_weibull['label']]})

        class_weibull['observation'] = class_weibull['observation'].reshape(-1, class_weibull['observation'].shape[0])
        class_weibull['observation'] = pd.DataFrame(class_weibull['observation'])
        class_weibull['observation'].columns = [f"x{i}"  for i in class_weibull['observation'].columns]

        final_df = pd.concat(list(class_weibull.values()), axis=1)
        old_names = final_df.columns
        first_names = ['shape', 'scale', 'params', 'label']
        last_names = [i for i in final_df.columns if i not in first_names]
        new_names = first_names + last_names
        final_df = final_df[new_names]
        weibull_dfs.append(final_df)

weibulls_df = pd.concat(weibull_dfs)

# ------------------- #
# -- add distances -- #
# ------------------- #
distances_ = distance_matrix(weibulls_df[last_names], X)

for i in range(distances_.shape[1]):
    weibulls_df[f"dist_{i}"] = distances_[:,i]

label_idx = np.where(weibulls_df.columns == 'label')[0][0]
dist_idx = np.where(weibulls_df.columns == 'dist_0')[0][0]

# ---------------------------- #
# -- Get weibull probabilities
# ---------------------------- #

probas_df = weibulls_df[['label']].copy()
dists_df = weibulls_df.iloc[:, dist_idx:].copy()
probas_df = dists_df*0
probas_df.columns = [f'proba_{col_}' for col_ in range(len(probas_df.columns))]

i_weibull = -1
while i_weibull < dists_df.shape[0]-1:
    i_weibull = i_weibull + 1

    weibull_pars_ = weibulls_df.iloc[i_weibull,]['params']
    weibull_pars_ = eval(weibull_pars_)

    dist_ = dists_df.iloc[i_weibull, :]
    proba_ = predict_weibul(x=dist_
                            , weibull_pars=weibull_pars_
                            )

    probas_df.iloc[i_weibull,:] = proba_


# ------------------------------------------------------- #
# -- get mean of the top `n_obs_to_fuse` probabilities -- #
# ------------------------------------------------------- #
prediction_dfs = {}
for label in labels:
    # label = labels[0]
    weibulls_df['label']
    aux_df = probas_df[weibulls_df['label'] == label]

    top_probs = []
    for observation in range(aux_df.shape[1]):
        # observation = 0
        probas = aux_df.iloc[:, observation]
        probas = probas.sort_values(ascending=False)
        probas = probas[:n_obs_to_fuse]
        probas = np.mean(probas)
        top_probs.append(probas)

    prediction_dfs[label] = top_probs


# --------------------- #
# -- predicted label -- #
# --------------------- #
len(prediction_dfs)
len(prediction_dfs[9])
prediction_df = pd.DataFrame(prediction_dfs)

