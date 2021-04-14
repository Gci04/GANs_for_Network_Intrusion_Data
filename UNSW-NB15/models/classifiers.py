import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support

from tabulate import tabulate

DISPLAY_PERFOMANCE = False

def __display_perfomance(ytrue, ypred,labels_mapping):
    classes = ["dos", "normal", "probe", "r2l", "u2r"]
    print("\nClass-wise Performance Report : ")
    print(classification_report(ytrue, ypred, labels = list(labels_mapping.values()),target_names=list(labels_mapping.keys())))

def __train_and_test(model,xtrain,ytrain,xtest,ytest,labels_mapping):

    model.fit(xtrain, ytrain)
    if DISPLAY_PERFOMANCE :
        predictions = model.predict(xtest)
        __display_perfomance(ytest,predictions,labels_mapping)
    return model

def decision_tree(xtrain, ytrain, xtest, ytest,labels_mapping):
    dt = DecisionTreeClassifier(max_depth=None)
    dt = __train_and_test(dt, xtrain, ytrain, xtest, ytest,labels_mapping)
    return dt

def random_forest(xtrain, ytrain, xtest, ytest,labels_mapping):
    rf = RandomForestClassifier(max_depth=None, n_jobs=-1) #n_estimators=13,
    rf = __train_and_test(rf, xtrain, ytrain, xtest, ytest,labels_mapping)
    return rf


def neural_network(xtrain, ytrain, xtest, ytest,labels_mapping, scaled = False):
    """
    First scale the data using StandardScaler if scaled == False
    """
    if not scaled :
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

    nn = MLPClassifier() #hidden_layer_sizes=30, alpha=0.0001, early_stopping=True
    nn = __train_and_test(nn, xtrain, ytrain, xtest, ytest,labels_mapping)
    return nn

def svm(xtrain, ytrain, xtest, ytest,labels_mapping, scaled = False):
    """
    First scale the data using StandardScaler if not scaled and maybe resample
    """
    if not scaled :
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

    svm = SVC(C=10, cache_size=1500, class_weight='balanced')
    svm = __train_and_test(svm, xtrain, ytrain, xtest, ytest,labels_mapping)
    return svm
