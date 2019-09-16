import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing import *
from classifiers import *
from utils import *
from matplotlib import pyplot as plt
# %matplotlib inline

train,test, label_mapping = get_data(encoding="Label")
data_cols = list(train.columns[ train.columns != 'label' ])
# x_train , x_test = preprocess(train,test,data_cols,"power_transform",True)
x_train , x_test = preprocess(train,test,data_cols,"Robust",True)
y_train = x_train.label.values
y_test = x_test.label.values
data_cols = list(x_train.columns[ x_train.columns != 'label' ])

#classification
# randf = random_forest(x_train, y_train, x_test, y_test)
# nn = neural_network(x_train[data_cols], y_train, x_test[data_cols], y_test,True)
# deci = decision_tree(x_train,y_train,x_test , y_test)
# catb = catBoost(x_train,y_train,x_test , y_test)
# nb = naive_bayes(x_train,y_train,x_test , y_test)

#Generative Adversarial Networks
att_ind = np.where(train.label == label_mapping["probe"])[0]

x = x_train[data_cols].values[att_ind]
# x = x_train[att_ind]

n_to_generate = 2000

rand_dim = 32
base_n_count = 100

combined_ep = 700
batch_size = 128 if len(x) > 128 else len(x)

ep_d = 1
ep_g = 1
learning_rate = 0.00001 #5e-5

arguments = [rand_dim, combined_ep, batch_size, ep_d,ep_g, learning_rate, base_n_count]
res = adversarial_training_GAN(arguments,x)

generated_samples = res["generator_model"].predict(np.random.normal(size=(n_to_generate,rand_dim)))

# x_train = np.vstack([x_train[data_cols].values,generated_samples])
# y_train = np.append(y_train,np.repeat(label_mapping["probe"],n_to_generate))

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
