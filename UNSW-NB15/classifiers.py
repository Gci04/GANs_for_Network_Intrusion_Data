import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support

from tabulate import tabulate

DISPLAY_PERFOMANCE = False
try:
    import catboost as cat
except ImportError:
    import pip
    pip.main(['install', '--user', 'catboost'])
    import catboost as cat

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

def catBoost(xtrain, ytrain, xtest, ytest,labels_mapping):
    cb = cat.CatBoostClassifier(verbose=0) #n_estimators=13,max_depth=5
    # {'depth': 10, 'iterations': 800, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
    cb = __train_and_test(cb, xtrain, ytrain, xtest, ytest,labels_mapping)
    return cb

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

def naive_bayes(xtrain, ytrain, xtest, ytest,labels_mapping):
    nb = GaussianNB()
    nb = __train_and_test(nb, xtrain, ytrain, xtest, ytest,labels_mapping)
    return nb

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

def kMeans(X_train, y_train, X_test, y_test, labels_mapping, scaled = False):
    pass
