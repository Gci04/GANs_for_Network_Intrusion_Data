import numpy as np
import pickle, os, random
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from tqdm import tqdm

from sklearn.decomposition import PCA

from utils import preprocessing
from utils import utils

# from models import cgan
import models.classifiers as clf

def main():
    #Load Data & Preprocess
    train,test, label_mapping = preprocessing.get_data(encoding="Label")
    data_cols = list(train.columns[ train.columns != 'label' ])

    #Remove contant values with a threshold
    to_drop = preprocessing.get_contant_featues(train,data_cols,threshold=0.995)

    train.drop(to_drop, axis=1,inplace=True)
    test.drop(to_drop, axis=1,inplace=True)

    #Normalize row-wise the data (to unit norm) and scale column-wise
    data_cols = list(train.columns[train.columns != 'label' ])
    train = preprocessing.normalize_data(train,data_cols)
    test = preprocessing.normalize_data(test,data_cols)
    x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",True)
    data_cols = list(x_train.columns[x_train.columns != 'label' ])

    train, test = None, None
    y_train = x_train.label.values
    y_test = x_test.label.values

    att_ind = np.where(x_train.label != label_mapping["normal"])[0]
    for_test = np.where(x_test.label != label_mapping["normal"])[0]

    del label_mapping["normal"]
    clf.DISPLAY_PERFOMANCE = False

    x = x_train[data_cols].values[att_ind] #x_train.query(f'label == {label_mapping["normal"]}').values
    y = y_train[att_ind]
    x_train, y_train = None, None
    print('Data read and preprocess : [DONE]')

    matplotlib_plots(x,y,label_mapping,'pca')
    matplotlib_plots(x,y,label_mapping,'tsne')
    plot_kl_div()

def plotly_plots(x,y,plot_type='pca'):
    fig = go.Figure()

    if plot_type == 'tsne':
        tsne = TSNE(n_components=components, verbose=1, perplexity=40, n_iter=300,n_jobs=-1)
        embedded_x = tsne.fit_transform(x)
    else:
        pca = PCA(n_components=components).fit(x)
        embedded_x = pca.transform(x)

    embedded_df = pd.DataFrame(embedded_x,columns=[f'Principal component {i+1}' for i in range(2) ])
    embedded_df["label"] = y
    colors = ['#7f7f7f','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#bcbd22','#17becf','#1f77b4']
    for y_label in np.unique(y):
        mask = np.where(y == y_label)[0]
        xs = embedded_x[mask,0]
        ys = embedded_x[mask,1]
        sz = np.ones(len(ys)) * 30
        fig.add_trace(go.Scatter(x=xs, y=ys,mode='markers',
                    marker=go.scatter.Marker(size=sz,opacity=0.6,colorscale="Viridis"),
                    name=f'label {y_label}')
                    )
    fig.show()

def plot_kl_div():
    filePath = './logs/CGAN_32_4_2000_128_1_1_relu_sgd_00005_27.pickle'
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
    ax.plot(n, kl,label='KL',linewidth=2,color="black")
    # ax.legend(loc=0,prop={'size': 10})
    ax.set_ylabel('KL-Divergence',fontsize=11.0)
    ax.set_xlabel('Epoch',fontsize=11.0)
    ax.tick_params(labelsize=11)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('./imgs/nsl_kl_div_plot.png', dpi = 350, bbox_inches='tight')
    fig.savefig('./imgs/nsl_kl_div_plot.pdf', bbox_inches='tight')
    fig.savefig('./imgs/nsl_kl_div_plot.eps',format='eps')

    # plt.show()

def matplotlib_plots(x,y,label_mapping,plot_type='pca'):

    components = 2
    if plot_type == 'tsne':
        tsne = TSNE(n_components=components, init='pca', verbose=1, perplexity=40, n_iter=300,n_jobs=-1,random_state=111)
        embedded_x = tsne.fit_transform(x)
    else:
        pca = PCA(n_components=components).fit(x)
        embedded_x = pca.transform(x)

    color = {0:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c', 4:'#d62728'}
    names = {j:i for i,j in label_mapping.items()}

    fig, ax = plt.subplots()
    colors_map = ['#9467bd','#1f77b4','#ff7f0e','#2ca02c','#d62728','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    embedded_x = np.hstack([embedded_x,y.reshape(-1,1)])

    color_ploting = [color.get(i) for i in embedded_x[:,2]]
    ax.scatter(embedded_x[:,0], embedded_x[:,1], c=color_ploting, s=150, alpha=0.7, edgecolors='#FFFFFF', linewidths = 0.5)

    for k, v in color.items():
        mask = np.where(k == embedded_x[:,2])[0]
        xs = embedded_x[mask,0]
        ys = embedded_x[mask,1]
        ax.scatter(xs[:5], ys[:5], c=[color.get(k,'#9467bd')]*5, s=150, alpha=0.7, edgecolors='#FFFFFF', linewidths = 0.5, label=names.get(k,'none'))

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

if __name__ == '__main__':
    main()
