import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import preprocessing
from classifiers import *
from utils import *
from matplotlib import pyplot as plt

train,test, label_mapping = preprocessing.get_data()
x_train,y_train = train.drop(["label","attack_cat"],axis=1),train.attack_cat.values
x_test , y_test =  test.drop(["label","attack_cat"],axis=1),test.attack_cat.values
randf = random_forest(x_train,y_train,x_test , y_test)
deci = decision_tree(x_train,y_train,x_test , y_test)
x_train.head()
# #Gans
# generate r2l attacks samples
x = x_train.iloc[np.where(y_train == 9)[0]]
data_cols = list(x.columns)
label_cols = ['attack_cat']

rand_dim = 32
base_n_count = 128

combined_ep = 700
batch_size = 128

ep_d = 1
ep_g = 2
learning_rate = 5e-5

x = StandardScaler().fit_transform(x)

arguments = [rand_dim,combined_ep,batch_size,ep_d,ep_g,learning_rate,base_n_count]
res = adversarial_training_GAN(arguments,x,data_cols,label_cols)

print(res["combined_model"].predict(np.random.normal(size=(3,32))))
plt.plot(np.arange(len(res["disc_loss_generated"])),res["disc_loss_generated"])
plt.title("UNSW-NS15 Combined model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
