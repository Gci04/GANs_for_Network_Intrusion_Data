import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import preprocessing
from classifiers import *
from utils import *
from matplotlib import pyplot as plt
# %matplotlib inline

train,test, label_mapping = preprocessing.get_data()
x_train,y_train = train.drop("label",axis=1),train.label.values
x_test , y_test =  test.drop("label",axis=1),test.label.values
# randf = random_forest(x_train,y_train,x_test , y_test)
# deci = decision_tree(x_train,y_train,x_test , y_test)
print(label_mapping)
#Gans
# generate r2l attacks samples
x = train.iloc[np.where(train.label == 0)[0]]
data_cols = list(x.columns[ x.columns != 'label' ])
label_cols = ['label']

rand_dim = 32
base_n_count = 128

combined_ep = 700
batch_size = 128

ep_d = 1
ep_g = 2
learning_rate = 5e-5

x = StandardScaler().fit_transform(x.drop('label',axis=1))

arguments = [rand_dim,combined_ep,batch_size,ep_d,ep_g,learning_rate,base_n_count]
res = adversarial_training_GAN(arguments,x,data_cols,label_cols)

print(res["combined_model"].predict(np.random.normal(size=(3,32))))

#plot the loss
plt.plot(np.arange(len(res["disc_loss_generated"])),res["disc_loss_generated"])
plt.title("NLS-KDD99 Combined model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
