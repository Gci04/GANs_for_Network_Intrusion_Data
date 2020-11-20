import numpy as np
import os, sys
import pickle

from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def plot_data(real,fake):
    """
    Visualization of the real and generated data to visualize the distribution differences.

    Parameters:
    ------------
    real : ndarray
        samples of the real data
    fake : ndarray
        samples of the generated data

    Return Value : None
    """
    fake_pca = PCA(n_components=2)
    fake_data = fake_pca.fit_transform(fake)

    real_pca = PCA(n_components=2)
    real_data = real_pca.fit_transform(real)

    _ , ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].scatter( real_data[:,0], real_data[:,1],color="g")
    ax[1].scatter( fake_data[:,0], fake_data[:,1],color="r")
    # ax[0].scatter( real[:,0], real[:,5],color="g")
    # ax[1].scatter( fake[:,0], fake[:,5],color="r")

    ax[0].set_title('Real')
    ax[1].set_title('Generated')

    ax[1].set_xlim(ax[0].get_xlim()), ax[1].set_ylim(ax[0].get_ylim())

    plt.show(block=False)
    plt.pause(3)
    plt.close()

def plot_distributions(real_dist,generated_dist):
    """Plot top and bottom 3 based on the KL-divergence value"""

    kl_values = np.sum(np.where(real_dist != 0, real_dist * np.log(real_dist/generated_dist),0),axis=1)
    top_3 = np.argsort(kl_values)[:3]
    bottom = np.argsort(kl_values)[-3:]

    tot_features = top_3.tolist() + bottom.tolist()
    sns.set(rc={'figure.figsize':(12,8)},font_scale=1.3)
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle('Distributions of Features')

    for ax, feature, name in zip(axes.flatten(), tot_features , tot_features):
        sns.distplot(real_dist[feature], hist = False, kde = True, ax = ax,
                     kde_kws = {'shade': True, 'linewidth': 3}, label = "Real")
        sns.distplot(generated_dist[feature], hist = False, kde = True,ax = ax,
                     kde_kws = {'shade': True, 'linewidth': 3}, label = "Fake")

        ax.set(title=f'Feature : {name}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show(block=False)
    plt.pause(5)
    plt.close()

def classifierAccuracy(X,g_z,n_samples):
    """
    Check how good are the generated samples using a catboost classifier. 50% accuracy
    implies that the real samples are identical and the classifier can not differentiate.
    The classifier is trained with 0.5 n_real + 1/2 n_fake samples.

    Parameters:
    -----------
    X : ndarray
        The real samples
    g_z : ndarray
        The fake samples produced by a generator

    Rerurn Value:
    -------------
    accuracy : float32
    """
    y_fake = np.ones(n_samples)
    y_true = np.zeros(n_samples)

    n = n_samples//2
    x_train = np.vstack((X[:n,:],g_z[:n,:]))
    y_train = np.hstack((np.zeros(n),np.ones(n))) # fake : 1, real : 0

    x_test = np.vstack((X[n:,:],g_z[n:,:]))
    y_test = np.hstack((np.zeros(n),np.ones(n_samples - n)))

    catb = cat.CatBoostClassifier(verbose=0,n_estimators=13,max_depth=5)
    catb.fit(x_train,y_train)

    pred = catb.predict(x_test)
    accuracy = accuracy_score(y_test,pred)

    return np.round(accuracy,decimals=3)

def modelAccuracy(gen_pred,real_pred):
    """calculates the discriminator's accuracy on real and generated samples
    :param gen_pred : predictions for generated samples
    :type gen_pred : ndarray (numpy)
    :param real_pred : predictions for real samples
    :type real_pred : ndarray (numpy)
    :return : accuracy_on_generated , accuracy_on_generated
    :rtype : float
    """
    gen_pred = np.array([1.0 if i > 0.5 else 0.0 for i in gen_pred])
    gen_true = np.zeros(len(gen_pred))

    real_pred = np.array([1.0 if i > 0.5 else 0.0 for i in real_pred])
    real_true = np.ones(len(gen_pred))

    # print('Discriminator accuracy on Fake : {}, Real : {}'.format(accuracy_score(gen_pred,gen_true),accuracy_score(real_pred,real_true)))
    return accuracy_score(gen_pred,gen_true)*100, accuracy_score(real_pred,real_true)*100

def PlotPCA(n_components,X,y,label_mapping,save_dir="./nsl_pca_plots"):
    """Visualise classters of data x, clusters labels y. PCA for reducing x to 2D"""
    names = {j:i for i,j in label_mapping.items()}
    pca = PCA(n_components=n_components).fit(X)
    embedded_x = pca.transform(X)
    color = {0:"red", 2:"blue", 3:"black", 4:"green"}

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, ax in enumerate(zip(axes.flatten(),np.unique(y))):
        mask = np.where(y == ax[1])[0]
        ax[0].scatter(embedded_x[mask,0], embedded_x[mask,1], c=color[ax[1]],label=f'{ax[1]}')
        #ax[0].legend(loc=0,prop={'size': 13})

    # fig.add_subplot(111, frameon=False)
    fig.text(0.5, 0.04, 'Principal component 1', va='center', ha='center' )
    fig.text(0.01, 0.5, 'Principal component 2', va='center', ha='center', rotation='vertical')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(save_dir,'combined_labels.png'), dpi = 300, bbox_inches='tight')
    plt.close('all')


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
        fig.savefig(os.path.join(save_dir,f'{names.get(label,"Non")}.png'), dpi = 300, bbox_inches='tight')
        plt.close('all')

    color = {0:"red", 2:"blue", 3:"black", 4:"green"}
    fig, ax = plt.subplots()
    for label in np.unique(y):
        mask = np.where(y == label)[0]

        ax.scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(label,"Non"),c = color[label])

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc=0,prop={'size': 13})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(save_dir,'nsl_all_labels.png'), dpi = 300, bbox_inches='tight')
    # fig.savefig(f'alllabel.pdf', bbox_inches='tight')
    # plt.savefig(f'alllabel.eps',format='eps')
    plt.close('all')

def plot_training_summary(filePath='',savePath='./imgs'):
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
    except Exception as ex :
        print(ex)
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

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    plt.savefig(os.path.join(savePath,figname+'.png'),dpi = 300)
    plt.close('all')

def load_pretrained_classifiers(dir = "./trained_classifiers"):
    assert os.path.exists(dir), f'{dir} : [does not exists]'

    res = {}
    for file in os.listdir(dir):
        file_full_path = os.path.join(dir, file)
        if os.path.isfile(file_full_path) and file.endswith(".pickle"):
            with open(file_full_path,"rb") as f:
                clf = pickle.load(f)
                res[clf.__class__.__name__] = clf

    if len(res) > 0: print("pretrained classifiers load : [DONE]")
    return res

def save_classifiers(clfs, dir = "./trained_classifiers"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for classifier in clfs:
        with open(os.path.join(dir ,f"{classifier.__class__.__name__}.pickle"), "wb") as f:
            pickle.dump(classifier,f)
    print("classifiers Save : [DONE]")
