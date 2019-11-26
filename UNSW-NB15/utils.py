import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import defaultdict
from classifiers import *

from tabulate import tabulate

def compare_classifiers(x_old, y_old, x_test, y_test, gan_generator, label_mapping, models,folds=3):
    """Compares the perfomace of models using recall, precision & fscore. Dumps the results in .txt file"""

    perfomace_results = defaultdict(defaultdict(dict).copy)

    for i in range(folds):
        print(f"Fold number : {i+1}")
        a = [0,1,2,7,8,9]
        labels = np.random.choice(a,(26000,1),p=[0.15,0.05,0.1,0.1,0.3,0.3],replace=True)
        generated_x = gan_generator.generate_data(labels)

        new_trainx = np.vstack([x_old,generated_x])
        new_y = np.append(y_old,labels)

        randf = random_forest(new_trainx, new_y, x_test, y_test,label_mapping)
        nn = neural_network(new_trainx, new_y, x_test, y_test,label_mapping,True)
        deci = decision_tree(new_trainx, new_y, x_test, y_test,label_mapping)
        nb = naive_bayes(new_trainx, new_y, x_test, y_test,label_mapping)

        for estimator in [randf,deci,nn,nb] :
            name = estimator.__class__.__name__
            pred = estimator.predict(x_test)
            precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred,labels=list(label_mapping.values()))
            perfomace_results[name][i]["precision"] = precision.tolist()
            perfomace_results[name][i]["recall"] = recall.tolist()
            perfomace_results[name][i]["fscore"] = fscore.tolist()

    for estimator in perfomace_results.keys():
        tempdf = pd.DataFrame.from_dict(perfomace_results[estimator][1])

        pred = models[estimator].predict(x_test)
        precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred,labels=list(label_mapping.values()))
        before_balance = pd.DataFrame(data=np.stack([precision,recall,fscore]).T,columns=list(tempdf.columns))

        for i in range(1,folds):
            tempdf += pd.DataFrame.from_dict(perfomace_results[estimator][i])

        tempdf = (tempdf/folds) - before_balance
        with open('UNSWperformance.txt', 'a') as outputfile:
            outputfile.write("\n"+estimator+"\n")
            print(tabulate(tempdf, headers='keys', tablefmt='psql'), file=outputfile)
