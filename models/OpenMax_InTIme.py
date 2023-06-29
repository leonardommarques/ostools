import os
import sys
import numpy as np
import pandas as pd
# import random
# import math
# import pickle
# from tqdm import tqdm
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Input, Conv1D, MaxPooling1D, \
# Dropout, BatchNormalization, Activation, Concatenate, Masking, GlobalAveragePooling1D, \
# Conv1DTranspose, Reshape, PReLU
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.regularizers import l2
# from keras.utils.np_utils import to_categorical
# from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
# from tslearn.datasets import UCR_UEA_datasets
# from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
# from sktime.utils.data_io import load_from_tsfile_to_dataframe
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.utils import class_weight
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as stats
from scipy.stats import weibull_min
# from scipy.special import softmax
# import scipy.spatial.distance as distance
# import scipy.optimize as optimize
import warnings
warnings.filterwarnings("ignore")

# ---------------------------- #
# -- OpenMax
# ---------------------------- #

def calc_distances(activations, mav):
    """
    Returns the distances between the activations of correctly classified samples and MAV of the given class
    """
    distances = np.empty(len(activations))
    for i in range(len(activations)):
        distances[i] = np.linalg.norm(activations[i] - mav)
    return distances



def fit_weibull(distances_all, n_classes):
    """
    Returns one Weibull model (set of parameters) for each class
    """
    weibull_models = []
    for i in range(n_classes):
        shape, loc, scale = weibull_min.fit(distances_all[i])
        weibull_models.append([shape, loc, scale])

    return weibull_models


