import numpy as np
import pickle, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from utils import preprocessing
from utils import utils

from models import cgan
import models.classifiers as clf

from tensorflow.keras.models import load_model

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

    utils.PlotPCA(2,x,y,label_mapping)
    print('PCA plot : [DONE]')
    #Load Ml classifiers
    ml_classifiers = utils.load_pretrained_classifiers()
    print('pretrained ml classifiers loaded : [OK]')

    #load pretrained GAN generator model
    model = load_model('./trained_generator/gen.h5')

    # Evaluate classifiers performance after balancing data with GAN or SMOTE
    clf.compare(x,y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping, ml_classifiers ,cv=5) #CGAN
    # for smoteMethod in ["ADASYN","SMOTEENN","BorderlineSMOTE", "SMOTE"]:
    #     clf.compare(x,y, x_test[data_cols].values[for_test], y_test[for_test], smoteMethod, label_mapping, ml_classifiers ,cv=5) #SMOTE

if __name__ == '__main__':
    main()
