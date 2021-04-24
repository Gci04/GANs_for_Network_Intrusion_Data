import numpy as np
import os, sys, pickle

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

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
