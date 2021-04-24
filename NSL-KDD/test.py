import numpy as np
import pickle, os, torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import preprocessing
from utils import utils

from models import cgan
import models.classifiers as clf

def main():
    #Load Data & Preprocess
    train, test, label_mapping = preprocessing.get_data(encoding="Label")
    data_cols = list(train.columns[ train.columns != 'label' ])

    _ , x_test, data_cols = preprocessing.preprocess(train,test,"Robust",True)

    train, test = None, None
    y_test = x_test.label.values
    x_test.drop("label", inplace=True, axis=1).values
    
    clf.DISPLAY_PERFOMANCE = False

    print('Data read and preprocess : [DONE]')

    #Load Ml classifiers
    # ml_classifiers = utils.load_pretrained_classifiers()
    # print('pretrained ml classifiers loaded : [OK]')
    #
    # #load pretrained GAN generator model
    # model = load_model('./trained_generator/gen.h5')
    #
    # # Evaluate classifiers performance after balancing data with GAN or SMOTE
    # clf.compare(x,y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping, ml_classifiers ,cv=5) #CGAN
    # # for smoteMethod in ["ADASYN","SMOTEENN","BorderlineSMOTE", "SMOTE"]:
    # #     clf.compare(x,y, x_test[data_cols].values[for_test], y_test[for_test], smoteMethod, label_mapping, ml_classifiers ,cv=5) #SMOTE

if __name__ == '__main__':
    main()
