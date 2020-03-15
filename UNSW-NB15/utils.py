import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from collections import defaultdict
from classifiers import *
from time import time
import pickle, os

from imblearn.under_sampling import RandomUnderSampler
from tabulate import tabulate
from preprocessing import normalize_data

def compare_classifiers(x_old, y_old, x_test, y_test, gan_generator, label_mapping, models,cv=3):
    """Compares the perfomace of models using recall, precision & fscore. Dumps the results in .txt file"""

    perfomace_results = defaultdict(defaultdict(dict).copy)

    #do downsapling
    rus = RandomUnderSampler(sampling_strategy = {label_mapping["Generic"]:20000,label_mapping["Exploits"]:20000},random_state=42)
    x_old , y_old = rus.fit_resample(x_old,y_old)

    a = {'Analysis':6000, 'Backdoor':6000, 'DoS':3000, 'Exploits':0,'Fuzzers':2000,\
    'Generic':0, 'Reconnaissance':5000, 'Shellcode':2000, 'Worms':300}
    temp = np.array(list(a.values()))
    p = temp/temp.sum()

    for i in range(cv):
        print(f"Cross validation number : {i+1}")
        labels = np.random.choice(list(label_mapping.values()),(temp.sum(),1),p=p,replace=True)
        generated_x = normalize_data(gan_generator.generate_data(labels),None)

        new_trainx = np.vstack([x_old,generated_x])
        new_y = np.append(y_old,labels)

        randf = random_forest(new_trainx, new_y, x_test, y_test,label_mapping)
        nn = neural_network(new_trainx, new_y, x_test, y_test,label_mapping,True)
        deci = decision_tree(new_trainx, new_y, x_test, y_test,label_mapping)
        # nb = naive_bayes(new_trainx, new_y, x_test, y_test,label_mapping)
        sVmclf = svm(new_trainx, new_y, x_test, y_test,label_mapping,True)

        for estimator in [randf,deci,nn,sVmclf] :
            name = estimator.__class__.__name__
            pred = estimator.predict(x_test)
            precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred,labels=list(label_mapping.values()))
            perfomace_results[name][i]["precision"] = precision.tolist()
            perfomace_results[name][i]["recall"] = recall.tolist()
            perfomace_results[name][i]["fscore"] = fscore.tolist()
            perfomace_results[name][i]["weighted_f1"] = [f1_score(y_test,pred,labels=list(label_mapping.values()),average='weighted')] * len(label_mapping)

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

def save_classifiers(clfs, dir = "./trained_classifiers"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for classifier in clfs:
        with open(dir + f"/{classifier.__class__.__name__}.pickle", "wb") as f:
            pickle.dump(classifier,f)
    print("classifiers Save : [DONE]")


def load_pretrained_classifiers(dir = "./trained_classifiers"):
    if not os.path.exists(dir):
        print(dir + " : [does not exists]")
        exit(1)
    else:
        res = {}
        for file in os.listdir(dir):
            file_full_path = os.path.join(dir, file)
            if os.path.isfile(file_full_path) and file_full_path[-7:] == ".pickle":
                with open(file_full_path,"rb") as f:
                    clf = pickle.load(f)
                    res[clf.__class__.__name__] = clf

        if len(res) > 0: print("pretrained classifiers load : [DONE]")
    return res
