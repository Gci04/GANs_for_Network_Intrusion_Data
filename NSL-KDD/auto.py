import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import preprocessing, pandas_profiling

from keras.layers import Input, Dense,Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf 
from keras import optimizers, regularizers, backend as K

train,test, label_mapping = preprocessing.get_data(encoding="Label")

data_cols = list(train.columns[ train.columns != 'label' ])
x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",False)
# train.shape

def AutoEncoder(Xtrain):
    input_dim = Xtrain.shape[1]
    latent_space_size = 12
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation="tanh")(input_)
    layer_2 = Dense(50, activation="tanh")(layer_1)
    layer_3 = Dense(25, activation="tanh")(layer_2)

    encoding = Dense(latent_space_size,activation=None)(layer_3)

    layer_5 = Dense(25, activation="tanh")(encoding)
    layer_6 = Dense(50, activation="tanh")(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim,activation=None)(layer_7)

    autoencoder = Model(inputs=input_ , outputs=decoded)
    # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer="adam")

    # Fit autoencoder

    autoencoder.fit(Xtrain, Xtrain,epochs=10,validation_split=0.1 ,batch_size=100,shuffle=True,verbose=1)

    #create dimension reducer
    dim_reducer = Model(inputs = input_, outputs = encoding)

    return autoencoder , dim_reducer

ae , dmr = AutoEncoder(x_train.drop('label',axis=1).values)
ae.summary()


newX = dmr.predict(x_train.drop('label',axis=1).values)
newdf = pd.DataFrame(data = newX, columns= [f'col_{i}' for i in range(12)])
# newdf.head()
profile = pandas_profiling.ProfileReport(newdf)
profile.to_file(outputfile="NSLKDD_reduced_report.html")

# pandas_profiling.ProfileReport(newdf)
