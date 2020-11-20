import numpy as np
import pickle, os
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE

# import seaborn as sns
# sns.set_style("darkgrid")

from sklearn.decomposition import PCA

from utils import preprocessing
from utils import utils

def main():
    print("Loading data [Started]")
    train,test, label_mapping = preprocessing.get_data()
    data_cols = list(train.drop(["label","attack_cat"],axis=1).columns)
    train = utils.normalize_data(train,data_cols)
    test = utils.normalize_data(test,data_cols)
    train , test = preprocessing.preprocess(train,test,data_cols,"Robust",True)

    x_train,y_train = train.drop(["label","attack_cat"],axis=1),train.attack_cat.values
    x_test , y_test =  test.drop(["label","attack_cat"],axis=1),test.attack_cat.values
    train,test = None, None

    data_cols = list(x_train.columns)

    to_drop = preprocessing.get_contant_featues(x_train,data_cols,threshold=0.99)
    print("get_contant_featues : [DONE]")
    x_train.drop(to_drop, axis=1,inplace=True)
    x_test.drop(to_drop, axis=1,inplace=True)
    data_cols = list(x_train.columns)
    print("Preprocessing data [DONE]")

    #filter out normal data points
    att_ind = np.where(y_train != label_mapping["Normal"])[0]
    for_test = np.where(y_test != label_mapping["Normal"])[0]

    del label_mapping["Normal"] #remove Normal netwok traffic from data
    x = x_train[data_cols].values[att_ind]
    y = y_train[att_ind]

    matplotlib_plots(x,y,label_mapping,'pca')
    matplotlib_plots(x,y,label_mapping,'tsne')
    plot_kl_div()

def matplotlib_plots(x,y,label_mapping,plot_type='pca'):
    components = 2
    fig, ax = plt.subplots()

    if plot_type == 'tsne':
        tsne = TSNE(n_components=components, init='pca', verbose=1, perplexity=40, n_iter=300,n_jobs=-1,random_state=111)
        embedded_x = tsne.fit_transform(x)
    else:
        pca = PCA(n_components=components).fit(x)
        embedded_x = pca.transform(x)

    names = {j:i for i,j in label_mapping.items()}
    # colors_map = ['#9467bd','#1f77b4','#ff7f0e','#2ca02c','#d62728','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    colors_map = {0:"red", 2:"blue", 3:"black", 4:"green", 5:"teal", 6:"magenta", 7:"grey", 8:"purple",9:"brown",1:"orange"}
    embedded_x = np.hstack([embedded_x,y.reshape(-1,1)])

    color_ploting = [colors_map.get(i) for i in embedded_x[:,2]]
    ax.scatter(embedded_x[:,0], embedded_x[:,1], c=color_ploting, s=150, alpha=0.7, edgecolors='#FFFFFF', linewidths = 0.5)

    for k in np.unique(y):
        mask = np.where(k == embedded_x[:,2])[0]
        xs = embedded_x[mask,0]
        ys = embedded_x[mask,1]
        ax.scatter(xs[:5], ys[:5], c=[colors_map.get(k,'red')]*5, s=150, alpha=0.7, edgecolors='#FFFFFF', linewidths = 0.5, label=names.get(k,'none'))

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=11)

    x_axis_name = 'Principal component 1' if plot_type == 'pca' else 't-SNE dimension 1'
    y_axis_name = 'Principal component 2' if plot_type == 'pca' else 't-SNE dimension 2'

    fig.text(0.5, 0.04, x_axis_name, va='center', ha='center', fontsize=11)
    fig.text(0.005, 0.5, y_axis_name, va='center', ha='center', rotation='vertical', fontsize=11)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend(bbox_to_anchor=(1.04,1),prop={'size': 11})
    fig.savefig(f'./imgs/{plot_type}.png', dpi = 350, bbox_inches='tight')
    fig.savefig(f'./imgs/{plot_type}.pdf', bbox_inches='tight')
    # fig.savefig(f'./imgs/{plot_type}.eps',format='eps')
    # plt.show()

def plot_kl_div():
    filePath = './logs/CGAN_32_4_6000_128_1_1_relu_sgd_00005_27.pickle'
    assert os.path.isfile(filePath) , f'{filePath} does not exist'
    try:
        with open(filePath, 'rb') as f:
            x = pickle.load(f)
    except Exception as ex :
        print(ex)
        print(f'could not open {filePath}')
        exit(1)
    d_l = np.array(x['discriminator_loss']).ravel()
    kl = np.array(x["kl_divergence"]).ravel()
    n = np.arange(0,(len(kl)*10),10)
    print(f"kl_divergence len {len(kl)}")

    fig, ax = plt.subplots()
    ax.plot(n[:300], kl[:300],label='KL',linewidth=2,color="black")
    # ax.legend(loc=0,prop={'size': 10})
    ax.set_ylabel('KL-Divergence',fontsize=11.0)
    ax.set_xlabel('Epoch',fontsize=11.0)
    ax.tick_params(labelsize=11)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('./imgs/unsw_kl_div_plot.png', dpi = 350, bbox_inches='tight')
    fig.savefig('./imgs/unsw_kl_div_plot.png', dpi = 350, bbox_inches='tight')
    # fig.savefig('./imgs/unsw_kl_div_plot.eps',format='eps')
    # plt.show()

if __name__ == '__main__':
    main()
