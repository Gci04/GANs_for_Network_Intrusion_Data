import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import preprocessing
import utils
import classifiers as clf
import cgan, VannilaGan, wgan
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

#-------------------- Load Data ----------------------------#
train,test, label_mapping = preprocessing.get_data(encoding="Label")
data_cols = list(train.columns[ train.columns != 'label' ])
x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",True)

y_train = x_train.label.values
y_test = x_test.label.values

data_cols = list(x_train.columns[ x_train.columns != 'label' ])

to_drop = preprocessing.get_contant_featues(x_train,data_cols)
x_train.drop(to_drop, axis=1,inplace=True)
x_test.drop(to_drop, axis=1,inplace=True)

data_cols = list(x_train.columns[ x_train.columns != 'label' ])

#---------------------classification ------------------------#
# randf = clf.random_forest(x_train, y_train, x_test, y_test)
# nn = clf.neural_network(x_train[data_cols], y_train, x_test[data_cols], y_test,True)
# deci = clf.decision_tree(x_train,y_train,x_test , y_test)
# catb = clf.catBoost(x_train,y_train,x_test , y_test)
# nb = clf.naive_bayes(x_train,y_train,x_test , y_test)

#---------------Generative Adversarial Networks -------------#
att_ind = np.where(x_train.label != label_mapping["normal"])[0]
x = x_train[data_cols].values[att_ind]
y = y_train[att_ind]

#---------------------Set GAN parameters--------------------#
rand_dim = 32
base_n_count = 50
combined_ep = 100
batch_size = 128 if len(x) > 128 else len(x)
ep_d = 1
ep_g = 1
learning_rate = 0.001 #5e-5
Optimizer = 'sgd'
activation = 'tanh'

#--------------------Define & Train GANS-----------------------#
#
# arguments = [rand_dim, combined_ep, batch_size, ep_d,ep_g, learning_rate, base_n_count]
# res = utils.adversarial_training_GAN(arguments,x)
#
# generated_samples = res["generator_model"].predict(np.random.normal(size=(n_to_generate,rand_dim)))
#
# x_train = np.vstack([x_train[data_cols].values,generated_samples])
# y_train = np.append(y_train,np.repeat(label_mapping["probe"],n_to_generate))
#
# #classification after upsampling
# # randf = random_forest(x_train,y_train,x_test , y_test)
# # nn = neural_network(x_train,y_train,x_test , y_test)
#
# #plot the loss
# plt.plot(np.arange(combined_ep),res["generator_loss"],label="Generator Loss")
# plt.plot(np.arange(combined_ep),res["discriminator_loss"],label="Discriminator Loss")
# plt.title("NLS-KDD99 GAN Losses")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()

#-------- Vannila GAN ---------#
# args = [rand_dim, combined_ep, batch_size,ep_d,ep_g, activation, Optimizer, learning_rate, base_n_count]
#
# vanilla_gan = VannilaGan.Vannila_GAN(args,x)
# vanilla_gan.train()
# vanilla_gan.save_model_componets()

#------- Conditional GAN ------#
args = [rand_dim, combined_ep, batch_size, ep_d,ep_g, activation, Optimizer, learning_rate, base_n_count]

cgan = cgan.CGAN(args,x_train.values,y_train.reshape(-1,1))
cgan.train()
cgan.dump_to_file()

#-------- Wasserstein GAN -------#
# ep_d = 5
# learning_rate = 0.0001
# args = [rand_dim, combined_ep, batch_size,ep_d,ep_g, learning_rate, base_n_count]
#
# wcgan = wgan.WGAN(args,x,y.reshape(-1,1))
# wcgan.train()
# wcgan.save_model_config()
