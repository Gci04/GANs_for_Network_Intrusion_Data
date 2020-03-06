import numpy as np
import pandas as pd

import preprocessing
import utils
import classifiers as clf
import cgan, VannilaGan, wgan

import matplotlib.pyplot as plt

def main(arguments):
    #-------------------- Load Data ----------------------------#
    train,test, label_mapping = preprocessing.get_data(encoding="Label")
    data_cols = list(train.columns[ train.columns != 'label' ])
    train = preprocessing.normalize_data(train,data_cols)
    test = preprocessing.normalize_data(test,data_cols)
    x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",True)

    train, test = None, None
    y_train = x_train.label.values
    y_test = x_test.label.values

    data_cols = list(x_train.columns[ x_train.columns != 'label' ])

    to_drop = preprocessing.get_contant_featues(x_train,data_cols,threshold=0.995)
    print(f"Constant Features : {to_drop}")
    x_train.drop(to_drop, axis=1,inplace=True)
    x_test.drop(to_drop, axis=1,inplace=True)

    data_cols = list(x_train.columns[ x_train.columns != 'label' ])
    clf.DISPLAY_PERFOMANCE = False

    #---------------------classification ------------------------#

    att_ind = np.where(x_train.label != label_mapping["normal"])[0]
    for_test = np.where(x_test.label != label_mapping["normal"])[0]
    # print(label_mapping)

    del label_mapping["normal"]
    print(label_mapping)
    svm = clf.svm(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping,False)
    randf = clf.random_forest(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    nn = clf.neural_network(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping,False)
    deci = clf.decision_tree(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping)

    #---------------Generative Adversarial Networks -------------#

    x = x_train[data_cols].values[att_ind] #x_train.query(f'label == {label_mapping["normal"]}').values
    y = y_train[att_ind]
    x_train, y_train = None, None

    #---------------------Set GAN parameters--------------------#

    args = arguments

    #Define & Train GANS
    #--------------------For Vannila Gan------------------#

    # print(args)
    # model = VannilaGan.Vannila_GAN(args,x)
    # model.train()
    # model.save_model_componets()

    #--------------------For cGAN-----------------------#

    print(args)
    model = cgan.CGAN(args,x,y.reshape(-1,1))
    model.train()
    model.dump_to_file()
    m = {"RandomForestClassifier":randf,"MLPClassifier":nn,"DecisionTreeClassifier":deci,"SVC":svm}
    clf.compare(x,y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping, m ,cv=5)


    # #--------------------For WCGAN-----------------------#


if __name__ == "__main__":
    # df = pd.read_csv("best_cgans.csv")
    # df["combined_ep"] = df['combined_ep']*2
    # for p in df.values:
    #     main(p.tolist())

    # a = [32, 500, 128, 1, 1, 'tanh', 'sgd', 0.0005, 27] #for vannilaGan
    a = [32, 4, 2000, 128, 1, 1, 'relu', 'sgd', 0.0005, 27] #for cGan
    main(a)
