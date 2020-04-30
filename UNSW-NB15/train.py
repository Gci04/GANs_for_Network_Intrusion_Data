import numpy as np
import pandas as pd
import os

from utils import preprocessing
from utils import utils

from models import cgan
import models.classifiers as clfrs

def main(arguments):
    #Load data & preprocess
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

    # train Ml classifiers
    print("Training classifiers : [Started]")
    clfrs.DISPLAY_PERFOMANCE = False
    randf = clfrs.random_forest(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    nn = clfrs.neural_network(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping,True)
    deci = clfrs.decision_tree(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    svmclf = clfrs.svm(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping,True)
    print("Training classifiers : [Finised]")

    utils.save_classifiers([randf,nn,deci,svmclf])
    print("Classifiers save to disk : [SUCCESSFUL]")

    #Define, Train & Save GAN
    print("GAN Training Starting ....")
    model = cgan.CGAN(arguments,x,y.reshape(-1,1))
    model.train()
    model.dump_to_file()
    print("GAN Training Finised!")

    #Plot GAN training logs
    gan_path = f"./logs/CGAN_{model.gan_name}.pickle"
    utils.plot_training_summary(gan_path,'./imgs')

if __name__ == '__main__':
    gan_params = [32, 4, 2000, 128, 1, 1, 'relu', 'sgd', 0.0005, 27]
    main(gan_params)
