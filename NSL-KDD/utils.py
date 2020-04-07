import numpy as np
import pandas as pd
import os, sys
import catboost as cat

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

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

def PlotPCA(n_components,X,y,label_mapping):
    """Visualise classters of data x, clusters labels y. PCA for reducing x to 2D"""
    names = {j:i for i,j in label_mapping.items()}
    pca = PCA(n_components=2).fit(X)
    embedded_x = pca.transform(X)
    color = {0:"red", 2:"blue", 3:"black", 4:"green"}

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
        fig.savefig(f'label{label}.pdf', bbox_inches='tight')
        plt.close('all') #plt.close(fig)

    color = {0:"red", 2:"blue", 3:"black", 4:"green"}
    fig, ax = plt.subplots()
    for label in np.unique(y):
        mask = np.where(y == label)[0]

        ax.scatter(embedded_x[mask,0], embedded_x[mask,1],label=names.get(label,"Non"),c = color[label])

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc=0,prop={'size': 13})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'alllabel.pdf', bbox_inches='tight')
    # plt.savefig(f'alllabel.eps',format='eps')
    plt.close('all') #plt.close(fig)
