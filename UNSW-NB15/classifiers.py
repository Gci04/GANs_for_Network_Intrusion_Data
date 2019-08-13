import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report,f1_score
import preprocessing

import matplotlib.pyplot as plt

try:
    import catboost as cat
except ImportError:
    import pip
    pip.main(['install', '--user', 'catboost'])
    import catboost

def __display_perfomance(ytrue, ypred):
    classes = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
    print("\nClass-wise Performance Report : ")
    print(classification_report(ytrue, ypred, target_names=classes))

def __train_and_test(model,xtrain,ytrain,xtest,ytest):

    print('-'*50)
    print('Training and Testing {}'.format(model.__class__.__name__))
    print('-'*50)
    model.fit(xtrain, ytrain)
    predictions = model.predict(xtest)
    __display_perfomance(ytest,predictions)
    return model

def decision_tree(xtrain, ytrain, xtest, ytest):
    dt = DecisionTreeClassifier(max_depth=None)
    dt = __train_and_test(dt, xtrain, ytrain, xtest, ytest)

def random_forest(xtrain, ytrain, xtest, ytest):
    rf = RandomForestClassifier(n_estimators=13, max_depth=None, n_jobs=-1)
    rf = __train_and_test(rf, xtrain, ytrain, xtest, ytest)
    return rf

def catBoost(xtrain, ytrain, xtest, ytest):
    cb = catboost.CatBoostClassifier(iterations=100)
    cb = __train_and_test(cb, xtrain, ytrain, xtest, ytest)
    return cb

def neural_network(xtrain, ytrain, xtest, ytest):
    """
    First scale the data using StandardScaler
    """
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    nn = MLPClassifier(hidden_layer_sizes=30, alpha=0.0001, early_stopping=True)
    nn = __train_and_test(nn, xtrain, ytrain, xtest, ytest)
    return nn

def naive_bayes(xtrain, ytrain, xtest, ytest):
    nb = GaussianNB()
    nb = __train_and_test(nb, xtrain, ytrain, xtest, ytest)
    return nb

def svm(xtrain, ytrain, xtest, ytest):
    """
    First scale the data using StandardScaler and maybe resample
    """
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    svm = SVC(C=10, cache_size=1500, class_weight='balanced')
    svm = __train_and_test(svm, xtrain, ytrain, xtest, ytest)
    return svm

def kMeans(X_train, y_train, X_test, y_test):
    pass
