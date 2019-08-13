import pandas as pd
import numpy as np
import sys, os , warnings
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from category_encoders import *

warnings.filterwarnings('ignore')

# data_folder = "../Data/UNSW-NB15"
data_folder = "/Users/gcinizwedlamini/Documents/GitHub/AnomalyDetectionNslKdd/UNSW-NB15/dataset/part_training_testing_set"

def get_data(encoding = "Label"):

    #drop id feature
    train = pd.read_csv(data_folder+"/UNSW_NB15_testing-set.csv",usecols=range(1,45))
    test = pd.read_csv(data_folder+"/UNSW_NB15_training-set.csv",usecols=range(1,45))

    categorical_features = ["proto", "service", "state"]

    le = LabelEncoder()
    le.fit(train.attack_cat)
    label_mapping = {l: i for i, l in enumerate(le.classes_)}

    train['attack_cat'] = le.transform(train.attack_cat)
    test['attack_cat'] = le.transform(test.attack_cat)

    if encoding == "OneHot":

        nTrain = train.shape[0]

        combined = pd.get_dummies(pd.concat((train,test),axis=0), columns=categorical_features)

        train = combined.iloc[:nTrain]
        test = combined.iloc[nTrain:]

    if encoding == 'Hashing':
        enc = HashingEncoder(cols=categorical_features)
        train = enc.fit_transform(train,train.label)
        test = enc.transform(test)

    if encoding == 'Label':
        enc = OrdinalEncoder(cols=categorical_features)
        train = enc.fit_transform(train,train.label)
        test = enc.transform(test)

    if encoding == 'LeaveOneOut' :
        enc = LeaveOneOutEncoder(cols=categorical_features)
        train = enc.fit_transform(train,train.label)
        test = enc.transform(test)
    if encoding == "catboost":
        enc = CatBoostEncoder(cols=categorical_features)
        train = enc.fit_transform(train,train.label)
        test = enc.transform(test)

    return train, test, label_mapping
