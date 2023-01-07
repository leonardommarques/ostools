import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#### Class hyperparameters ###
self = OpenMax_Bendale(
    trained_model = model
    , tail_size=10
    , classes_to_revise=-1
    , distance_function=scipy.spatial.distance.euclidean
)

self.fit(X_train, Y_train)

### -- ### fit: def fit(...)
self
X = X_train.copy()
y=Y_train.copy()
predictions=None

self