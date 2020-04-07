import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from collections import defaultdict
from classifiers import *
from time import time
import pickle, os
from sklearn.svm import LinearSVC

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE
from tabulate import tabulate
from preprocessing import normalize_data

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

def compare_classifiers(x_old, y_old, x_test, y_test, gan_generator, label_mapping, models,cv=3):
    """Compares the perfomace of models using recall, precision & fscore. Dumps the results in .txt file"""

    a = {'Analysis':6000, 'Backdoor':6000, 'DoS':3000, 'Exploits':0,'Fuzzers':2000,\
    'Generic':0, 'Reconnaissance':5000, 'Shellcode':2000, 'Worms':300}
    temp = np.array(list(a.values()))
    p = temp/temp.sum()

    perfomace_results = defaultdict(defaultdict(dict).copy)
    val , count = np.unique(y_old,return_counts =True)
    current_gt = dict(zip(val , count))
    up_sampling_strategy = {label_mapping.get(j) : i+current_gt.get(label_mapping.get(j)) for j, i in a.items() if i > 0}
    # up_sampling_strategy = {}
    print(up_sampling_strategy)

    start_t = time()
    if gan_generator == None:
        print("Using SMOTE")

        sm = SVMSMOTE(sampling_strategy = up_sampling_strategy,svm_estimator=SVC(C=10, cache_size=1500, kernel = "linear",class_weight='balanced'),n_jobs=-1)
        # sub_x, sub_y = subsample(x=x_old,y=y_old,label_mapping=label_mapping,size=25000)
        # sm.fit(sub_x,sub_y)

    elapsed_time = time() - start_t
    print(f"Time taken : {elapsed_time}")

    rus = RandomUnderSampler(sampling_strategy = {label_mapping["Generic"]:20000,label_mapping["Exploits"]:20000},random_state=42)
    x_old , y_old = rus.fit_resample(x_old,y_old)

    for i in range(cv):
        print(f"Cross validation number : {i+1}")

        if gan_generator != None:
            labels = np.random.choice(list(label_mapping.values()),(temp.sum(),1),p=p,replace=True)
            generated_x = normalize_data(gan_generator.generate_data(labels),None)
            new_trainx = np.vstack([x_old,generated_x])
            new_y = np.append(y_old,labels)
        else :
            start_t = time()
            sub_x, sub_y = subsample(x=x_old,y=y_old,size=25000)
            new_trainx, new_y = sm.fit_resample(sub_x, sub_y)
            elapsed_time = time() - start_t
            print(f"Time taken : {elapsed_time}")

        randf = random_forest(new_trainx, new_y, x_test, y_test,label_mapping)
        nn = neural_network(new_trainx, new_y, x_test, y_test,label_mapping,True)
        deci = decision_tree(new_trainx, new_y, x_test, y_test,label_mapping)
        #new_trainx, new_y = subsample(x=new_trainx,y=new_y,size=25000)
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

        # pred = models[estimator].predict(x_test)
        # precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred,labels=list(label_mapping.values()))
        # weighted_f1 = [f1_score(y_test,pred,labels=list(label_mapping.values()),average='weighted')] * len(label_mapping)
        # before_balance = pd.DataFrame(data=np.stack([precision,recall,fscore,weighted_f1]).T,columns=list(tempdf.columns),index=list(label_mapping.values()))

        # tempdf = tempdf #- before_balance
        for i in range(1,cv):
            to_append = pd.DataFrame.from_dict(perfomace_results[estimator][i])
            to_append.index = list(label_mapping.values())
            tempdf = tempdf.append(to_append)
            #tempdf = tempdf.append(to_append - before_balance)

        tempdf["class"] = tempdf.index.map({y:x for x,y in label_mapping.items()})
        tempdf = tempdf.groupby("class").agg({'precision': ['mean', 'std'], 'recall': ['mean', 'std'], 'fscore': ['mean', 'std'], 'weighted_f1' : ['mean', 'std']})
        tempdf.columns = [f"{i[0]}_{i[1]}" for i in tempdf.columns]

        with open(f'results/performance{t}_{cv}validations.txt', 'a') as outputfile:
            outputfile.write("\n"+estimator+"\n")
            print(tabulate(tempdf, headers='keys', tablefmt='psql'), file=outputfile)

    name = gan_generator.gan_name if gan_generator != None else "SMOTE"
    with open(f'results/performance{t}_{cv}validations.txt', 'a') as outputfile:
        outputfile.write(name)

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

def subsample(x, y, size=25000):
    """Sample data from x """
    new_counts = np.ceil(np.bincount(y)/y.shape[0] * size)
    data = np.hstack((x, y.reshape((y.shape[0], 1))))
    class_wise_arrays = []

    for cls in range(10):
        class_wise_arrays.append(data[y == cls])

    class_wise_random_indices = []
    for array, count in zip(class_wise_arrays, new_counts):
        class_wise_random_indices.append(np.random.choice(array.shape[0], int(count), replace=False))

    data_small = np.vstack(tuple([array[index, :] for array, index in zip(class_wise_arrays, class_wise_random_indices)]))
    x_small = data_small[:, :-1]
    y_small = data_small[:, -1]

    return x_small, y_small.astype("int")

def PlotPCA(n_components,X,y,label_mapping):
    """Visualise classters of data x, clusters labels y. PCA for reducing x to 2D"""
    names = {j:i for i,j in label_mapping.items()}
    pca = PCA(n_components=2).fit(X)
    embedded_x = pca.transform(X)
    color = {0:"red", 2:"blue", 3:"black", 4:"green", 5:"teal", 6:"magenta", 7:"grey", 8:"purple",9:"brown",1:"orange"}

    for label in np.unique(y):
        mask = np.where(y == label)[0]
        fig = plt.figure()
        #fig.suptitle(f'Label : {label}', fontsize=15,fontweight="bold")

        plt.scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(label,"Non"),c=color[label])

        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.legend(loc=0,prop={'size': 13})
        # plt.savefig(f'label{label}.eps',format='eps')
        fig.savefig(f'./pcaplots/{names.get(label,"Non")}.pdf', bbox_inches='tight')
        plt.close('all') #plt.close(fig)

    # color = {0:"red", 2:"blue", 3:"black", 4:"green"}
    fig, ax = plt.subplots()
    for label in np.unique(y):
        mask = np.where(y == label)[0]

        ax.scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(label,"Non"),c = color[label])

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc=0,prop={'size': 13})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'./pcaplots/alllabel.pdf', bbox_inches='tight')
    # plt.savefig(f'alllabel.eps',format='eps')
    plt.close('all') #plt.close(fig)
