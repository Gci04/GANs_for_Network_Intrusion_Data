import pandas as pd
import numpy as np
import sys, os , warnings
import pandas_profiling
from sklearn.preprocessing import LabelEncoder, StandardScaler ,MinMaxScaler,RobustScaler, PowerTransformer, normalize
from category_encoders import *

warnings.filterwarnings('ignore')

data_folder = "../Data/NSL-KDD"

def get_data(encoding = 'Label', data_folder = "../Data/NSL-KDD"):
    """
    Retrive Train and Test data
    """

    train = pd.read_csv(data_folder+"/KDDTrain.csv")
    test = pd.read_csv(data_folder+"/KDDTest.csv")

    le = LabelEncoder()
    le.fit(train.label)
    label_mapping = {l: i for i, l in enumerate(le.classes_)}

    train['label'] = le.transform(train.label)
    test['label'] = le.transform(test.label)

    if encoding == "OneHot":
        nTrain = train.shape[0]
        combined = pd.get_dummies(pd.concat((train,test),axis=0), columns=["protocol_type","service","flag"])
        train = combined.iloc[:nTrain]
        test = combined.iloc[nTrain:]

    if encoding == 'Hashing':
        enc = HashingEncoder(cols=["protocol_type","service","flag"])
        train = enc.fit_transform(train,train.label)
        test = enc.transform(test)

    if encoding == 'Label':
        enc = OrdinalEncoder(cols=["protocol_type","service","flag"])
        train = enc.fit_transform(train,train.label)
        test = enc.transform(test)

    if encoding == 'LeaveOneOut' :
        enc = LeaveOneOutEncoder(cols=["protocol_type","service","flag"])
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
        # profile = pandas_profiling.ProfileReport(x_train)
        # to_drop = profile.get_rejected_variables()
        to_drop = ['dst_host_srv_serror_rate','num_root','rerror_rate',
                    'serror_rate','srv_rerror_rate','srv_serror_rate']
        x_train.drop(to_drop,axis=1,inplace=True)
        x_test.drop(to_drop,axis=1,inplace=True)
        data_cols = list(x_train.columns[ x_train.columns != 'label' ])

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

def remove_outliers(X):
    Q1 = X.drop('label',axis=1).quantile(0.1)
    Q3 = X.drop('label',axis=1).quantile(0.99)
    IQR = Q3 - Q1

    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)

    return X[mask]

def normalize_data(X,data_cols):
    """Scale input vectors individually to unit norm (vector length)"""
    if  data_cols is None:
        return normalize(X)
    else :
        X[data_cols] = normalize(X[data_cols])
        return X
