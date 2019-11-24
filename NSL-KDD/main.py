import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import preprocessing
import utils, gc, itertools
import classifiers as clf
import cgan, VannilaGan, wgan
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# gc.disable()
#-------------------- Load Data ----------------------------#
train,test, label_mapping = preprocessing.get_data(encoding="Label")
data_cols = list(train.columns[ train.columns != 'label' ])
x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",False)
train, test = None, None
y_train = x_train.label.values
y_test = x_test.label.values

# data_cols = ["service","flag","src_bytes","dst_bytes","root_shell","is_host_login","serror_rate","same_srv_rate","diff_srv_rate","dst_host_srv_diff_host_rate","label"]
# x_train = x_train[data_cols]
# x_test = x_test[data_cols]

data_cols = list(x_train.columns[ x_train.columns != 'label' ])

to_drop = preprocessing.get_contant_featues(x_train,data_cols)
x_train.drop(to_drop, axis=1,inplace=True)
x_test.drop(to_drop, axis=1,inplace=True)

data_cols = list(x_train.columns[ x_train.columns != 'label' ])

#---------------------classification ------------------------#
# randf = clf.random_forest(x_train[data_cols], y_train, x_test[data_cols], y_test)
# nn = clf.neural_network(x_train[data_cols], y_train, x_test[data_cols], y_test,True)
# deci = clf.decision_tree(x_train[data_cols],y_train,x_test[data_cols] , y_test)
# catb = clf.catBoost(x_train[data_cols],y_train,x_test[data_cols] , y_test)
# nb = clf.naive_bayes(x_train,y_train,x_test , y_test)

#---------------Generative Adversarial Networks -------------#
att_ind = np.where(x_train.label != label_mapping["normal"])[0]
x = x_train[data_cols].values[att_ind]
y = y_train[att_ind]
# x_train, y_train = None, None
#---------------------Set GAN parameters--------------------#
rand_dim = 32
base_n_count = 27
n_layers = 4
combined_ep = 500
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

labels =np.random.choice([0,2,3,4],(1000,1),p=[0.1,0.2,0.3,0.4],replace=True)
print(np.unique(labels,return_counts=True))
generated_x = model.generate_data(labels)

new_trainx = np.vstack([x_train[data_cols].values,generated_x])
new_y = np.append(y_train,labels)
#
# randf = clf.random_forest(new_trainx, new_y, x_test[data_cols], y_test)
# nn = clf.neural_network(new_trainx, new_y, x_test[data_cols], y_test,True)
# deci = clf.decision_tree(new_trainx,new_y,x_test[data_cols] , y_test)
# catb = clf.catBoost(new_trainx,new_y,x_test[data_cols] , y_test)

# op =  ["sgd", "adam", "Adagrad","Adadelta","Adamax","Nadam","RMSprop"]
# lr = [0.1,0.01,0.001,0.0001]
# params = list(itertools.product([rand_dim],[n_layers], [100,200,500,1000], [batch_size], [1],[1], ['relu','tanh'], op, lr, [base_n_count]))
# print('Total parameters sets : {}'.format(len(params)))
# gc.collect()
# gc.enable()
# for arg in params:
#     args = list(arg)
#     model = cgan.CGAN(args,x_train.values,y_train.reshape(-1,1))
#     model.train()
#     model.dump_to_file()

#-------- Wasserstein GAN -------#
# ep_d = 5
# learning_rate = 0.0001
# args = [rand_dim, combined_ep, batch_size,ep_d,ep_g, learning_rate, base_n_count]
#
# wcgan = wgan.WGAN(args,x,y.reshape(-1,1))
# wcgan.train()
# wcgan.save_model_config()
# labels =np.random.choice([0,2,3,4],(1000,1),p=[0.1,0.2,0.3,0.4],replace=True)
# wcgan.generate_data(labels)
