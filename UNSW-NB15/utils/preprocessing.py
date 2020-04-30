import pandas as pd
import numpy as np
import sys, os , warnings , pandas_profiling
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler ,MinMaxScaler,RobustScaler, PowerTransformer, normalize
from category_encoders import *

warnings.filterwarnings('ignore')

data_folder = "../Data/UNSW-NB15"

def get_data(encoding = "Label"):

    #drop id feature
    train = pd.read_csv(data_folder+"/UNSW_NB15_testing-set.csv",usecols=range(1,45))
    test = pd.read_csv(data_folder+"/UNSW_NB15_training-set.csv",usecols=range(1,45))

    categorical_features = ["proto","state","service"]

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

def preprocess(x_train, x_test, data_cols, preprocessor = "StandardScaler",reject_features=False):
    """
    Scale and transform data with an option to remove highly correlated features
    """
    if reject_features :
        to_drop =['ct_srv_dst', 'ct_srv_src', 'dloss', 'dpkts', 'is_ftp_login', 'sloss', 'spkts', 'swin']
        #profile = pandas_profiling.ProfileReport(x_train)
        #to_drop = profile.get_rejected_variables(0.95)
        x_train.drop(to_drop,axis=1,inplace=True)
        x_test.drop(to_drop,axis=1,inplace=True)
        data_cols = list(x_train.columns)[:-2]

    if preprocessor == "MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train[data_cols] = scaler.fit_transform(x_train[data_cols])
        x_test[data_cols] = scaler.transform(x_test[data_cols])
        return x_train, x_test

    if preprocessor == "Robust":
        scaler = RobustScaler(quantile_range=(0.1, 99.9))
        x_train[data_cols] = scaler.fit_transform(x_train[data_cols])
        x_test[data_cols] = scaler.transform(x_test[data_cols])
        return x_train, x_test

    if preprocessor == "power_transform":
        pt = PowerTransformer(method="yeo-johnson")
        x_train[data_cols] = pt.fit_transform(x_train[data_cols])
        x_test[data_cols] = pt.transform(x_test[data_cols])
        return x_train, x_test

    else :
        scaler = StandardScaler()
        x_train[data_cols] = scaler.fit_transform(x_train[data_cols])
        x_test[data_cols] = scaler.transform(x_test[data_cols])
        return x_train, x_test

def get_contant_featues(X,data_cols,threshold=0.995):
    """
    Finds columns with contant value

    Parameters:
    ----------
    X : pandas DataFrame, shape = [n_samples, n_features]
        Dataset to be analyzed
    data_cols : List, array-like
        feature names of the input data X
    threshold : Float
        threshold to determine if a feature has contant value

    Return
    ------
    result : List , array-like
        list of features having a contant value in Data X
     """
    result = []
    for col in data_cols:
        val, counts = np.unique(X[col],return_counts=True)
        v = counts[0]/counts.sum()
        if v > threshold:
            result.append(col)

    return result
