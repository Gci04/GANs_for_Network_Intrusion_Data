import numpy as np
import pandas as pd
import os, sys, matplotlib
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras import optimizers
from scipy.stats import norm

class CGAN():
    def __init__(self,arguments,X,y):
        [self.rand_noise_dim, self.tot_epochs, self.batch_size,self.D_epochs,\
         self.G_epochs, self.learning_rate, self.min_num_neurones,self.learning_rate] = arguments

        self.X_train = X
        self.y_train = y

        self.label_dim = y.shape[1]
        self.x_data_dim = X.shape[1]

        self.discriminator_input_dim = self.x_data_dim + self.label_dim
        self.generator_input_dim = self.rand_noise_dim + self.label_dim


        opt = optimizers.Adam(lr=self.learning_rate)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=opt,metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        combined_input = Input(shape=(self.generator_input_dim,))

        combined_output = self.discriminator(self.generator(combined_input))
        self.combined = Model(inputs = gan_input, outputs = combined_output)
        self.combined.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Stacked generator and discriminator


    def build_generator(self):

        model = tf.keras.models.Sequential(name="Generator")

        model.add(Dense(self.min_num_neurones, activation='relu',input_dim = self.generator_input_dim ))
        model.add(Dense(self.min_num_neurones*2, activation='relu'))
        model.add(Dense(self.min_num_neurones*4, activation='tanh'))
        model.add(Dense(self.data_dim))

        model.summary()


        return model #Model([noise, label], img)

    def build_discriminator(self):

        model = tf.keras.models.Sequential(name='Discriminator')

        model.add(Dense(self.min_num_neurones*2, activation='relu',input_dim = self.discriminator_input_dim ))
        model.add(Dense(self.min_num_neurones, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return None

    def __get_batch_idx(self):
        batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=False)
        return batch_ix


    def train(self, epochs, batch_size=128, sample_interval=50):

        #print(self.generator.summary())
        #print(self.discriminator.summary())
        #print(self.combined.summary())

        # Adversarial ground truths
        real_labels = np.ones((self.batch_size, 1))
        fake_labes = np.zeros((self.batch_size, 1))
        # Adversarial ground truths with noise
        #real_labels = np.random.uniform(low=0.999, high=1.0, size=(self.batch_size,1))
        #fake_labes = np.random.uniform(low=0, high=0.00001, size=(self.batch_size,1))

        for epoch in range(epochs):

            #  Train Discriminator
            for i in range(self.D_epochs):

                idx = self.__get_batch_idx()

                x, labels = self.X_train[idx], self.y_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.rand_noise_dim))

                # Generate a half batch of new images
                generated_x = self.generator.predict([noise, labels])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([x, labels], real_labels)
                d_loss_fake = self.discriminator.train_on_batch([generated_x, labels], fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator
            for j in range(self.G_epochs):
                # Condition on labels
                sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], real_labels)

            # Plot the progress
            print ("Epoch : {} [D loss: {0:.4f}, acc.: {0:.4f}] [G loss: {0:.4f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss))
