import numpy as np
import pickle, os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# %matplotlib inline
# a = os.listdir('logs')

( _, _, filenames) = next(os.walk('UNSW-NB15/logs'))


def plot_summary(d_l, g_l,acc_r,acc_g, m =''):
    n = np.arange(len(d_l))
    title = 'Loss and Accuracy plot'+'\n'+ m
    title = title.replace('.pickle','')
    fig, axs = plt.subplots(2,figsize=(19.20,10.80))

    axs[0].set_title(title,fontsize=20.0,fontweight="bold")
    axs[0].plot(n, g_l,label='Generator loss',linewidth=4)
    axs[0].plot(n, d_l,label='Discriminator loss',linewidth=4)
    axs[0].legend(loc=0, prop={'size': 20})
    axs[0].set_ylabel('Loss',fontsize=20.0,fontweight="bold")
    axs[0].tick_params(labelsize=20)
    axs[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False,labelsize=20)

    # axs[1].plot(n, acc,'r',label='Discriminator accuracy',linewidth=4)
    axs[1].plot(n, acc_g,label='Accuracy on Generated',linewidth=4)
    axs[1].plot(n, acc_r,label='Accuracy on Real',linewidth=4)
    axs[1].legend(loc=0,prop={'size': 20})
    axs[1].set_ylabel('Accuracy',fontsize=20.0,fontweight="bold")
    axs[1].set_xlabel('Ephoc',fontsize=20.0,fontweight="bold")
    axs[1].tick_params(labelsize=20)

    plt.tight_layout()
    if not os.path.exists("UNSW-NB15/imgs"):
        os.makedirs('UNSW-NB15/imgs')
    plt.savefig(f'UNSW-NB15/imgs/{m[:-7]}.png',dpi = 300)
    plt.close('all') #plt.close(fig)


for filename in filenames :
    with open('UNSW-NB15/logs/'+filename, 'rb') as f:
        x = pickle.load(f)

    d_l = np.array(x['discriminator_loss']).ravel()
    g_l = np.array(x['Generator_loss']).ravel()
    acc_history = np.array(x['acc_history'])
    acc = acc_history.sum(axis=1) * 0.5
    acc_real = acc_history[:,1]
    acc_gen = acc_history[:,0]

    plot_summary(d_l, g_l,acc_real,acc_gen,filename)
