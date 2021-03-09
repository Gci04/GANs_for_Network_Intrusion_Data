import numpy as np
import pickle, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from utils import preprocessing
from utils import utils

from models import cgan
import models.classifiers as clfrs

from tensorflow.keras.models import load_model

def main():
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

    #plot data PCA
    utils.PlotPCA(2,x,y,label_mapping)

    #Load pretrained ml classifiers
    ml_classifiers = utils.load_pretrained_classifiers()

    #Load trained GAN generator model
    model = load_model('./trained_generators/gen.h5')
    print("pretrained generator model load : [DONE]")

    #Genetare new data samples, fit ML models compare perfomance with ML models before data balancing
    utils.compare_classifiers(x,y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping, ml_classifiers ,cv=5)

    # test on SMOTE method
   for smoteMethod in ["ADASYN","SMOTEENN","BorderlineSMOTE", "SMOTE"]:
       utils.compare_classifiers(x,y, x_test[data_cols].values[for_test], y_test[for_test], smoteMethod, label_mapping, ml_classifiers ,cv=5)

if __name__ == '__main__':
    main()
