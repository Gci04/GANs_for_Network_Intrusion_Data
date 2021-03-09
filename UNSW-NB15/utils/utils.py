import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from collections import defaultdict
from models.classifiers import *
from time import time
import pickle, os
from sklearn.svm import LinearSVC

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN
from tabulate import tabulate

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style("darkgrid")

def compare_classifiers(x_old, y_old, x_test, y_test, data_generator, label_mapping, models,cv=3):
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
    nameUpsampler = "DGM"
    if isinstance(data_generator,str):
        print(f'Using : {data_generator}')
        up_sampling_strategy = {label_mapping["probe"]:14656, label_mapping["r2l"]:13995,label_mapping["u2r"]:10052}
        if data_generator == "ADASYN":
            sm = ADASYN(sampling_strategy = up_sampling_strategy,n_jobs=-1)
        elif data_generator == "SMOTEENN":
            sm = SMOTEENN(sampling_strategy = up_sampling_strategy,n_jobs=-1)
        elif data_generator == "BorderlineSMOTE" :
            sm = BorderlineSMOTE(sampling_strategy = up_sampling_strategy,n_jobs=-1)
        else:
            sm = SVMSMOTE(sampling_strategy = up_sampling_strategy,svm_estimator=SVC(C=10, cache_size=1500, class_weight='balanced'))

        sm.fit(x_old,y_old)
        nameUpsampler = type(sm).__name__

    elapsed_time = time() - start_t
    print(f"Time taken : {elapsed_time}")
    rus = RandomUnderSampler(sampling_strategy = {label_mapping["Generic"]:20000,label_mapping["Exploits"]:20000},random_state=42)
    x_old , y_old = rus.fit_resample(x_old,y_old)

    for i in range(cv):
        print(f"Cross validation number : {i+1}")

        if not isinstance(data_generator,str):
            labels = np.random.choice(list(label_mapping.values()),(temp.sum(),1),p=p,replace=True)
            rand_noise_dim = data_generator.input_shape[0][-1]
            noise = np.random.normal(0, 1, (len(labels), rand_noise_dim))
            #print([noise, labels])
            generated_x = normalize_data(data_generator.predict([noise, labels])[:,:-1],None)
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

        for i in range(1,cv):
            to_append = pd.DataFrame.from_dict(perfomace_results[estimator][i])
            to_append.index = list(label_mapping.values())
            tempdf = tempdf.append(to_append)

        tempdf["class"] = tempdf.index.map({y:x for x,y in label_mapping.items()})
        tempdf = tempdf.groupby("class").agg({'precision': ['mean', 'std'], 'recall': ['mean', 'std'], 'fscore': ['mean', 'std'], 'weighted_f1' : ['mean', 'std']})
        tempdf.columns = [f"{i[0]}_{i[1]}" for i in tempdf.columns]

        with open(f'./results/performance_{nameUpsampler}_{cv}validations.txt', 'a') as outputfile:
            outputfile.write("\n"+estimator+"\n")
            print(tabulate(tempdf, headers='keys', tablefmt='psql'), file=outputfile)

    with open(f'./results/performance_{nameUpsampler}_{cv}validations.txt', 'a') as outputfile:
        outputfile.write(nameUpsampler)

def save_classifiers(clfs, dir = "./trained_classifiers"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for classifier in clfs:
        with open(os.path.join(dir ,f"{classifier.__class__.__name__}.pickle"), "wb") as f:
            pickle.dump(classifier,f)
    print("classifiers Save : [DONE]")


def load_pretrained_classifiers(dir = "./trained_classifiers"):
    assert os.path.exists(dir), f'{dir} : [does not exists]'

    res = {}
    for file in os.listdir(dir):
        file_full_path = os.path.join(dir, file)
        if os.path.isfile(file_full_path) and file_full_path[-7:] == ".pickle":
            with open(file_full_path,"rb") as f:
                clf = pickle.load(f)
                res[clf.__class__.__name__] = clf

    if len(res) > 0: print("pretrained classifiers load : [DONE]")
    return res

def normalize_data(X,data_cols):
    """Scale input vectors individually to unit norm (vector length)"""
    if  data_cols is None:
        return normalize(X)
    else :
        X[data_cols] = normalize(X[data_cols])
        return X

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

def PlotPCA(n_components,X,y,label_mapping,save_dir="./unsw_pcaplots"):
    """Visualise classters of data x, clusters labels y. PCA for reducing x to 2D"""
    names = {j:i for i,j in label_mapping.items()}
    pca = PCA(n_components=2).fit(X)
    embedded_x = pca.transform(X)
    color = {0:"red", 2:"blue", 3:"black", 4:"green", 5:"teal", 6:"magenta", 7:"grey", 8:"purple",9:"brown",1:"orange"}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    for i, ax in enumerate(zip(axes.flatten(),np.unique(y))):
        mask = np.where(y == ax[1])[0]
        ax[0].scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(ax[1],"Non"), c=color[ax[1]])
        ax[0].legend(loc=0,prop=dict(size=7))

    # fig.add_subplot(111, frameon=False)
    fig.text(0.5, 0.01, 'Principal component 1', va='center', ha='center' )
    fig.text(0.01, 0.5, 'Principal component 2', va='center', ha='center', rotation='vertical')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,'unsw_combined_labels.png'), dpi = 300, bbox_inches='tight')
    plt.close('all')

    for label in np.unique(y):
        mask = np.where(y == label)[0]
        fig = plt.figure()
        plt.scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(label,"Non"),c=color[label])

        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.legend(loc=0,prop={'size': 13})
        # plt.savefig(f'label{label}.eps',format='eps')
        fig.savefig(os.path.join(save_dir,f'{names.get(label,"Non")}.png'), dpi = 300,bbox_inches='tight')
        plt.close('all')

    fig, ax = plt.subplots()
    for label in np.unique(y):
        mask = np.where(y == label)[0]
        ax.scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(label,"Non"),c = color[label])

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc=0,prop={'size': 13})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(save_dir,'unsw_all_attacks.png'), dpi = 300, bbox_inches='tight')
    # plt.savefig(save_dir+ '/unsw_all_attacks.eps',format='eps')
    plt.close('all')

def plot_training_summary(filePath='',savePath='.s/imgs'):
    """
    Plot and Save GAN training metrices i.e discriminator, generator loss & accuracy, KL-Divergence
    :param filePath : sting, path to file contatining GAN training logs file
    :param savePath : string, directory path to save resuting plot

    :return : None
    """
    assert os.path.isfile(filePath) , f'{filePath} does not exist'
    try:
        with open(filePath, 'rb') as f:
            x = pickle.load(f)
    except :
        print(f'could not open {filePath}')
        exit(1)

    d_l = np.array(x['discriminator_loss']).ravel()
    g_l = np.array(x['Generator_loss']).ravel()
    acc_history = np.array(x['acc_history'])
    acc = acc_history.sum(axis=1) * 0.5
    acc_real = acc_history[:,1]
    acc_gen = acc_history[:,0]
    kl = np.array(x["kl_divergence"]).ravel()

    n = np.arange(len(d_l))
    figname = os.path.split(filePath)[1].replace('.pickle','')
    title = 'Loss and Accuracy plot'+'\n'+ figname

    fig = plt.figure(figsize=(19.20,10.80))
    fig.suptitle(title, fontsize=15,fontweight="bold")

    axs1 = plt.subplot(222)
    # axs1.set_title(title,fontsize=5.0,fontweight="bold")
    axs1.plot(n, g_l,label='Generator loss',linewidth=3)
    axs1.plot(n, d_l,label='Discriminator loss',linewidth=3)
    axs1.legend(loc=0, prop={'size': 13})
    axs1.set_ylabel('Loss',fontsize=15.0,fontweight="bold")
    axs1.tick_params(labelsize=10)
    # axs1.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False,labelsize=20)

    # axs2.plot(n, acc,'r',label='Discriminator accuracy',linewidth=4)
    axs2 = plt.subplot(221)
    axs2.plot(n, acc_gen,label='Accuracy on Generated',linewidth=3)
    axs2.plot(n, acc_real,label='Accuracy on Real',linewidth=3)
    axs2.legend(loc=0,prop={'size': 13})
    axs2.set_ylabel('Accuracy',fontsize=15.0,fontweight="bold")
    # axs2.set_xlabel('Epoch',fontsize=15.0,fontweight="bold")
    axs2.tick_params(labelsize=10)

    axs3 = plt.subplot(212)
    n = np.arange(0,(len(kl)*10),10)

    axs3.plot(n, kl,label='KL',linewidth=3)
    axs3.legend(loc=0,prop={'size': 13})
    axs3.set_ylabel('KL-Divergence',fontsize=15.0,fontweight="bold")
    axs3.set_xlabel('Epoch',fontsize=15.0,fontweight="bold")
    axs3.tick_params(labelsize=10)

    # plt.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    plt.savefig(os.path.join(savePath,figname+'.png'),dpi = 300)
    plt.close('all') #plt.close(fig)
