import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import preprocessing
from classifiers import *
from utils import *
from matplotlib import pyplot as plt
# %matplotlib inline

train,test, label_mapping = preprocessing.get_data(encoding="Label")
x_train,y_train = train.drop("label",axis=1),train.label.values
x_test , y_test =  test.drop("label",axis=1),test.label.values

Scaler = StandardScaler()
x_train = Scaler.fit_transform(x_train)
x_test = Scaler.transform(x_test)
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
# x_train = min_max_scaler.fit_transform(x_train)
# x_test = min_max_scaler.transform(x_test)



#classification
# randf = random_forest(x_train, y_train, x_test, y_test)
# nn = neural_network(x_train, y_train, x_test, y_test, True)
# deci = decision_tree(x_train,y_train,x_test , y_test)
# catb = catBoost(x_train,y_train,x_test , y_test)
# nb = naive_bayes(x_train,y_train,x_test , y_test)

#Generative Adversarial Networks
att_ind = np.where(train.label == label_mapping["probe"])[0]

x = x_train[att_ind]
n_to_generate = 2000

rand_dim = 32
base_n_count = 50

combined_ep = 1000
batch_size = 128 if len(x) > 128 else len(x)

ep_d = 1
ep_g = 1
learning_rate = 5e-5

arguments = [rand_dim, combined_ep, batch_size, ep_d,ep_g, learning_rate, base_n_count]
res = adversarial_training_GAN(arguments,x)

generated_samples = res["generator_model"].predict(np.random.normal(size=(n_to_generate,rand_dim)))
x_train = np.vstack([x_train,generated_samples])
y_train = np.append(y_train,np.repeat(label_mapping["probe"],n_to_generate))

#classification after upsampling
# randf = random_forest(x_train,y_train,x_test , y_test)
# nn = neural_network(x_train,y_train,x_test , y_test)

#plot the loss
plt.plot(np.arange(combined_ep),res["generator_loss"],label="Generator Loss")
plt.plot(np.arange(combined_ep),res["discriminator_loss"],label="Discriminator Loss")
plt.title("NLS-KDD99 GAN Losses")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
