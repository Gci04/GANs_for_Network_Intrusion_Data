import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import preprocessing, cgan, utils
import classifiers as clfrs

def main(arguments):
    train,test, label_mapping = preprocessing.get_data()
    data_cols = list(train.drop(["label","attack_cat"],axis=1).columns)
    train = preprocessing.normalize_data(train,data_cols)
    test = preprocessing.normalize_data(test,data_cols)
    train , test = preprocessing.preprocess(train,test,data_cols,"Robust",True)
    print(label_mapping)
    x_train,y_train = train.drop(["label","attack_cat"],axis=1),train.attack_cat.values
    x_test , y_test =  test.drop(["label","attack_cat"],axis=1),test.attack_cat.values
    train,test = None, None

    data_cols = list(x_train.columns)

    to_drop = preprocessing.get_contant_featues(x_train,data_cols,threshold=0.99)
    print("get_contant_featues : [DONE]")
    x_train.drop(to_drop, axis=1,inplace=True)
    x_test.drop(to_drop, axis=1,inplace=True)
    data_cols = list(x_train.columns)

    clfrs.DISPLAY_PERFOMANCE = False

    #---------------------classification ------------------------#

    att_ind = np.where(y_train != label_mapping["Normal"])[0]
    for_test = np.where(y_test != label_mapping["Normal"])[0]

    del label_mapping["Normal"]
    x = x_train[data_cols].values[att_ind]
    y = y_train[att_ind]
    print(x.shape)

    pretrained_classifiers = True
    if not pretrained_classifiers :
        print("Training classifiers : [Started]")
        randf = clfrs.random_forest(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
        nn = clfrs.neural_network(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping,True)
        deci = clfrs.decision_tree(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
        svmclf = clfrs.svm(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping,True)

        m = {}
        for clf in [randf,nn,deci,svmclf]:
            m[clf.__class__.__name__] = clf

        utils.save_classifiers([randf,nn,deci,svmclf])
    else:
        m = utils.load_pretrained_classifiers()

    #---------------------CGAN Parameters set ------------------------#

    args = arguments

    #--------------------Define & Train GANS-----------------------#

    print(args)
    model = cgan.CGAN(args,x,y.reshape(-1,1))
    model.train()
    model.dump_to_file()

    # m = {"RandomForestClassifier":randf,"MLPClassifier":nn,"DecisionTreeClassifier":deci,"SVC":svmclf}
    # utils.compare_classifiers(x,y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping, m ,cv=3)

if __name__ == '__main__':
    # df = pd.read_csv("best_cgans.csv")
    # df["combined_ep"] = df['combined_ep']*2
    # for p in df.values:
        # print(p)
        # main(p.tolist())
    a = [32, 4,6000, 128 , 1, 1, 'relu', 'sgd', 0.0005, 27]
    main(a)
