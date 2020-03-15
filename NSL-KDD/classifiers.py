import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from collections import defaultdict

from tabulate import tabulate
from imblearn.under_sampling import RandomUnderSampler
from time import time
import os
from preprocessing import normalize_data

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

def kMeans(xtrain, ytrain, xtest, ytest,labels_mapping, scaled = True):
    """MiniBatchKMeans"""
    if not scaled :
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

    kmeans = MiniBatchKMeans(n_clusters=len(labels_mapping.values()), random_state=0)
    kmeans = __train_and_test(kmeans, xtrain, ytrain, xtest, ytest,labels_mapping)
    return kmeans

def compare(x_old, y_old, x_test, y_test, gan_generator, label_mapping, models,cv=3):

    if not os.path.exists("./results"):
        os.makedirs("results")

    perfomace_results = defaultdict(defaultdict(dict).copy)
    #do downsapling
    rus = RandomUnderSampler(sampling_strategy = {label_mapping["dos"]:20000},random_state=42)
    x_old , y_old = rus.fit_resample(x_old,y_old)

    for i in range(cv):
        print(f"Cross validation number  : {i+1}")
        labels =np.random.choice(list(label_mapping.values()),(26000,1),p=[0.0,0.115,0.5,0.385],replace=True)
        generated_x = gan_generator.generate_data(labels)

        new_trainx = np.vstack([x_old,generated_x])
        new_y = np.append(y_old,labels)

        randf = random_forest(new_trainx, new_y, x_test, y_test,label_mapping)
        nn = neural_network(new_trainx, new_y, x_test, y_test,label_mapping,True)
        deci = decision_tree(new_trainx, new_y, x_test, y_test,label_mapping)
        supvm = svm(new_trainx, new_y, x_test, y_test,label_mapping,True)
        # kms = kMeans(new_trainx, new_y, x_test, y_test,label_mapping,True)
        # nb = naive_bayes(new_trainx, new_y, x_test, y_test,label_mapping)

        for estimator in [randf,deci,nn,supvm] :
            name = estimator.__class__.__name__
            pred = estimator.predict(x_test)
            precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred,labels=[0,2,3,4])
            perfomace_results[name][i]["precision"] = precision.tolist()
            perfomace_results[name][i]["recall"] = recall.tolist()
            perfomace_results[name][i]["fscore"] = fscore.tolist()
            perfomace_results[name][i]["weighted_f1"] = [f1_score(y_test,pred,labels=[0,2,3,4],average='weighted')] * len(label_mapping)

    t = int(time())
    for estimator in perfomace_results.keys():
        tempdf = pd.DataFrame.from_dict(perfomace_results[estimator][0])
        tempdf.index = list(label_mapping.values())

        pred = models[estimator].predict(x_test)
        precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred,labels=list(label_mapping.values()))
        weighted_f1 = [f1_score(y_test,pred,labels=list(label_mapping.values()),average='weighted')] * len(label_mapping)
        before_balance = pd.DataFrame(data=np.stack([precision,recall,fscore,weighted_f1]).T,columns=list(tempdf.columns),index=list(label_mapping.values()))

        tempdf = tempdf - before_balance

        for i in range(1,cv):
            to_append = pd.DataFrame.from_dict(perfomace_results[estimator][i])
            to_append.index = list(label_mapping.values())
            tempdf = tempdf.append(to_append - before_balance)

        tempdf["class"] = tempdf.index.map({y:x for x,y in label_mapping.items()})
        tempdf = tempdf.groupby("class").agg({'recall': ['mean', 'std'], 'precision': ['mean', 'std'], 'fscore': ['mean', 'std'], 'weighted_f1' : ['mean', 'std']})
        tempdf.columns = [f"{i[0]}_{i[1]}" for i in tempdf.columns]

        with open(f'results/performance{t}_{cv}validations.txt', 'a') as outputfile:
            outputfile.write("\n"+estimator+"\n")
            print(tabulate(tempdf, headers='keys', tablefmt='psql'), file=outputfile)

    with open(f'results/performance{t}_{cv}validations.txt', 'a') as outputfile:
        outputfile.write(gan_generator.gan_name)
