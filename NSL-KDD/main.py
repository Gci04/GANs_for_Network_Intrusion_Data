import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

#classification
randf = random_forest(x_train, y_train, x_test, y_test)
nn = neural_network(x_train, y_train, x_test, y_test, True)
# deci = decision_tree(x_train,y_train,x_test , y_test)
# catb = catBoost(x_train,y_train,x_test , y_test)
# nb = naive_bayes(x_train,y_train,x_test , y_test)

#Generative Adversarial Networks
att_ind = np.where(train.label == label_mapping["u2r"])[0]

x = x_train[att_ind]
n_to_generate = 2000

rand_dim = 32
base_n_count = 128

combined_ep = 2000
batch_size = 128 if len(x) > 128 else len(x)

ep_d = 1
ep_g = 1
learning_rate = 5e-5

arguments = [rand_dim, combined_ep, batch_size, ep_d,ep_g, learning_rate, base_n_count]
res = adversarial_training_GAN(arguments,x)

generated_samples = res["generator_model"].predict(np.random.normal(size=(n_to_generate,rand_dim)))
x_train = np.vstack([x_train,generated_samples])
y_train = np.append(y_train,np.repeat(label_mapping["u2r"],n_to_generate))

#classification after upsampling
randf = random_forest(x_train,y_train,x_test , y_test)
nn = neural_network(x_train,y_train,x_test , y_test)

#plot the loss
plt.plot(np.arange(len(res["combined_loss"])),res["combined_loss"])
plt.title("NLS-KDD99 Combined model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
