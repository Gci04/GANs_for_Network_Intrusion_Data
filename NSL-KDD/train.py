import numpy as np
import pandas as pd

from utils import preprocessing
from utils import utils
from models import classifiers as clf
from models import cgan

import matplotlib.pyplot as plt

def main(arg):
    #-------------------- Load Data & Preprocess ---------------------#
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

    # # train Ml classifiers
    # print("Training classifiers : [Started]")
    # svm = clf.svm(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping,False)
    # randf = clf.random_forest(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    # nn = clf.neural_network(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping,False)
    # deci = clf.decision_tree(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    # print("Training classifiers : [Finished]")
    #
    # #Save ML trained models to disk models
    # utils.save_classifiers([svm,randf,nn,deci])
    # print("Classifiers save to disk : [SUCCESSFUL]")

    x = x_train[data_cols].values[att_ind] #x_train.query(f'label == {label_mapping["normal"]}').values
    y = y_train[att_ind]
    x_train, y_train = None, None

    #Define, Train & Save GAN
    print("GAN Training Starting ....")
    model = cgan.CGAN(arg,x,y.reshape(-1,1))
    model.train()
    print(model.generate_data(np.array([1,2,3])))
    # model.dump_to_file()
    # print("GAN Training & Save [SUCCESSFUL]")
    #
    # #Plot GAN training logs
    # gan_path = f"./logs/CGAN_{model.gan_name}.pickle"
    # utils.plot_training_summary(gan_path,'./imgs')

if __name__ == '__main__':
    # gan_params = [32, 500, 128, 1, 1, 'tanh', 'sgd', 0.0005, 27] #for vannilaGan
    gan_params = [32, 5,100, 256 , 1, 1, 'spocu', 'sgd', 0.00003, 5] #for cGan
    main(gan_params)
