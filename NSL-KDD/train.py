import numpy as np
import pandas as pd
import torch

from utils import preprocessing
from utils import utils
from models import classifiers as clf
from models import cgan

import matplotlib.pyplot as plt

def main(arg):
    #-------------------- Load Data & Preprocess ---------------------#
    train, test, label_mapping = preprocessing.get_data(encoding="Label")
    data_cols = list(train.columns[ train.columns != 'label' ])

    train, test, data_cols = preprocessing.preprocess(train,test,"Robust",True)

    y_test = test.label.values
    x_test = test.drop("label", axis=1)
    print(f"Test size {len(y_test)}")

    y_train = train.label.values
    x_train = train.drop("label", axis=1)
    print(f"Train size {len(y_train)}")

    train, test = None, None


if __name__ == '__main__':
    gan_params = {"noise_dim":32, "n_epochs":3000, "batch_size": 128, "lr":0.0001} #for cGan
    main(gan_params)
