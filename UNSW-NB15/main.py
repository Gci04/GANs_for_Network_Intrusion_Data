import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import preprocessing, cgan, utils
import classifiers as clf

train,test, label_mapping = preprocessing.get_data()
data_cols = list(train.drop(["label","attack_cat"],axis=1).columns)
train , test = preprocessing.preprocess(train,test,data_cols,"Robust",True)

x_train,y_train = train.drop(["label","attack_cat"],axis=1),train.attack_cat.values
x_test , y_test =  test.drop(["label","attack_cat"],axis=1),test.attack_cat.values
train,test = None, None

data_cols = list(x_train.columns)

# to_drop = preprocessing.get_contant_featues(x_train,data_cols,threshold=0.999)
# x_train.drop(to_drop, axis=1,inplace=True)
# x_test.drop(to_drop, axis=1,inplace=True)
# data_cols = list(x_train.columns)

clf.DISPLAY_PERFOMANCE = False

#---------------------classification ------------------------#

att_ind = np.where(y_train != label_mapping["Normal"])[0]
for_test = np.where(y_test != label_mapping["Normal"])[0]

del label_mapping["Normal"]
x = x_train[data_cols].values[att_ind]
y = y_train[att_ind]

randf = clf.random_forest(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
nn = clf.neural_network(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping,True)
deci = clf.decision_tree(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
catb = clf.catBoost(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
nb = clf.naive_bayes(x, y, x_test[data_cols].values[for_test], y_test[for_test],label_mapping)

#---------------------CGAN Parameters set ------------------------#

rand_dim = 32
base_n_count = 27
n_layers = 4
combined_ep = 500 #500
batch_size = 64 if len(x) > 128 else len(x)
ep_d , ep_g = 1, 1
learning_rate = 0.001 #5e-5
Optimizer = 'sgd'
activation = 'tanh'
args = [rand_dim,n_layers,combined_ep ,batch_size,ep_d,ep_g,activation,Optimizer,learning_rate,base_n_count]

#--------------------Define & Train GANS-----------------------#

model = cgan.CGAN(args,x,y.reshape(-1,1))
model.train()
model.dump_to_file()
#
m = {"RandomForestClassifier":randf,"MLPClassifier":nn,"DecisionTreeClassifier":deci,"GaussianNB":nb}
utils.compare_classifiers(x,y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping, m ,folds=3)
