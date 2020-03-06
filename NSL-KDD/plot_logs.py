import numpy as np
import pickle, os, sys
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def plot_summary(d_l, g_l,acc_r,acc_g, kl, m =''):
    n = np.arange(len(d_l))
    title = 'Loss, Accuracy & KL-Divergence plot'+'\n'+ m
    title = title.replace('.pickle','')

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
    axs2.plot(n, acc_g,label='Accuracy on Generated',linewidth=3)
    axs2.plot(n, acc_r,label='Accuracy on Real',linewidth=3)
    axs2.legend(loc=0,prop={'size': 13})
    axs2.set_ylabel('Accuracy',fontsize=15.0,fontweight="bold")
    # axs2.set_xlabel('Ephoc',fontsize=15.0,fontweight="bold")
    axs2.tick_params(labelsize=10)

    axs3 = plt.subplot(212)
    n = np.arange(0,(len(kl)*10),10)
    # print("N = {}, Real n = {}".format(len(n),len(kl)))
    axs3.plot(n, kl,label='KL',linewidth=3)
    axs3.legend(loc=0,prop={'size': 13})
    axs3.set_ylabel('KL-Divergence',fontsize=15.0,fontweight="bold")
    axs3.set_xlabel('Ephoc',fontsize=15.0,fontweight="bold")
    axs3.tick_params(labelsize=10)

    # plt.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'imgs/{m[:-7]}.png',dpi = 300)
    plt.close('all') #plt.close(fig)
    # plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1 :
        name = str(sys.argv[1]) #path to logs pickle
        print(name)
        with open( name, 'rb') as f:
            x = pickle.load(f)
        name = name.split("/")[-1]

        d_l = np.array(x['discriminator_loss']).ravel()
        g_l = np.array(x['Generator_loss']).ravel()
        acc_history = np.array(x['acc_history'])
        acc = acc_history.sum(axis=1) * 0.5
        acc_real = acc_history[:,1]
        acc_gen = acc_history[:,0]
        kl = np.array(x["kl_divergence"]).ravel()
        plot_summary(d_l, g_l,acc_real,acc_gen,kl,name)
        
    else:
        df = pd.read_csv("best_cgans.csv")
        # df["combined_ep"] = df['combined_ep']*2
        for p in df.values:
            # main(p.tolist())
            name = "logs/CGAN_" + '_'.join(str(e) for e in p.tolist()).replace(".","") + ".pickle"

            print(name)
            with open( name, 'rb') as f:
                x = pickle.load(f)
            name = name.split("/")[-1]

            d_l = np.array(x['discriminator_loss']).ravel()
            g_l = np.array(x['Generator_loss']).ravel()
            acc_history = np.array(x['acc_history'])
            acc = acc_history.sum(axis=1) * 0.5
            acc_real = acc_history[:,1]
            acc_gen = acc_history[:,0]
            kl = np.array(x["kl_divergence"]).ravel()
            plot_summary(d_l, g_l,acc_real,acc_gen,kl,name)
