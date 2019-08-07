import pandas as pd
import numpy as np
import sys, os , warnings
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from category_encoders import *

warnings.filterwarnings('ignore')

data_folder = "./Data/NSL-KDD"

def get_data():
    encoding = 'Label'
    features = None
    with open('features.txt', 'r') as f:
      features = f.read().split('\n')

    train_att = pd.read_csv('attacks_types.txt',sep=" ",header=None,index_col=0)
    attack_map = train_att.to_dict('dict')[1]

    train = pd.read_csv(data_folder+"/KDDTrain+.txt",usecols=range(42),names=features)
    test = pd.read_csv(data_folder+"/KDDTest+.txt",usecols=range(42),names=features)

    test["label"] = test["label"].apply(lambda x: attack_map.get(x,"Unknown"))
    train["label"] = train["label"].apply(lambda x: attack_map.get(x,"Unknown"))

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

    return train, test, label_mapping
