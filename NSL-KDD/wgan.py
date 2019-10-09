import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras import optimizers

from sklearn.metrics import accuracy_score
from collections import defaultdict
import os, pickle

class WGAN(object):
    """Wasserstein Generative Adversarial Network Class"""

    def __init__(self, args,X,y):
        [self.rand_noise_dim, self.tot_epochs, self.batch_size,self.D_epochs,\
         self.G_epochs, self.learning_rate, self.min_num_neurones] = args

        self.X_train = X
        self.y_train = y

        self.label_dim = y.shape[1]
        self.x_data_dim = X.shape[1]

        self.sgd = optimizers.SGD(lr=self.learning_rate)

        self.g_losses = []
        self.d_losses, self.disc_loss_real, self.disc_loss_generated = [], [], []

        self.__define_models()

        self.clip_value = 0.5

    def wasserstein_loss(self,y_true, y_pred):
        """define earth mover distance (wasserstein loss)"""
        return K.mean(y_true * y_pred)

    def define_generator(self,x,labels):
        """Build a Generator Model"""
        x = concatenate([x,labels])
        x = Dense(self.min_num_neurones*1, activation='relu')(x)
        x = Dense(self.min_num_neurones*2, activation='relu')(x)
        x = Dense(self.x_data_dim)(x)
        x = concatenate([x,labels])

        return x

    def define_critic(self,x):
        """Build a critic"""

        x = Dense(self.min_num_neurones*4, activation='tanh')(x)
        x = Dense(self.min_num_neurones*2, activation='tanh')(x)
        x = Dense(self.min_num_neurones, activation='tanh')(x)
        x = Dense(1, activation=None)(x)

        return x

    def __define_models(self):
        """Define Generator, Discriminator & combined model"""

        # Create & Compile generator
        generator_input = Input(shape=(self.rand_noise_dim,))
        labels_tensor = Input(shape=(self.label_dim,))
        generator_output = self.define_generator(generator_input, labels_tensor)

        self.generator = Model(inputs=[generator_input, labels_tensor], outputs=[generator_output], name='generator')

        # Create & Compile critic
        critic_model_input = Input(shape=(self.x_data_dim + self.label_dim,))
        critic_output = self.define_critic(critic_model_input)

        self.critic_network = Model(inputs=[critic_model_input],outputs=[critic_output],name='critic')
        self.critic_network.compile(loss=[self.wasserstein_loss], optimizer=self.sgd)

        # Build "frozen critic"
        frozen_critic = Model(inputs=[critic_model_input],outputs=[critic_output],name='frozen_critic')
        frozen_critic.trainable = False

        # Build & compile combined model from critic frozen weights + generator
        combined_output = frozen_critic(generator_output)
        self.combined = Model(inputs = [generator_input, labels_tensor],outputs = [combined_output],name='adversarial_model')
        self.combined.compile(loss=[self.wasserstein_loss], optimizer=self.sgd)

    def __get_batch_idx(self):
        """random selects batch_size samples indeces from training data"""
        batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=False)

        return batch_ix

    def train(self):
        # Adversarial ground truths
        real_labels = -np.ones((self.batch_size, 1))
        fake_labels = np.ones((self.batch_size, 1))

        for epoch in range(self.tot_epochs):
            #Train critic
            for i in range(self.D_epochs):

                idx = self.__get_batch_idx()
                x, labels = self.X_train[idx], self.y_train[idx]

                #Sample noise as generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.rand_noise_dim))

                #Generate a half batch of new images
                generated_x = self.generator.predict([noise, labels])

                #Train the critic
                d_loss_fake = self.critic_network.train_on_batch(generated_x, fake_labels)
                d_loss_real = self.critic_network.train_on_batch(np.concatenate((x,labels),axis=1), real_labels)
                d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

                for layer in self.critic_network.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight, -self.clip_value, self.clip_value) for weight in weights]
                    layer.set_weights(weights)

            self.disc_loss_real.append(d_loss_real)
            self.disc_loss_generated.append(d_loss_fake)
            self.d_losses.append(d_loss)

            #Train Generator (generator in combined model is trainable while discrimnator is frozen)
            for j in range(self.G_epochs):
                #Condition on labels
                # sampled_labels = np.random.randint(1, 5, self.batch_size).reshape(-1, 1)
                sampled_labels = np.random.choice([0,2,3,4],(self.batch_size,1), replace=True)

                #Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], real_labels)
                self.g_losses.append(g_loss)

            if epoch % 10 == 0:
                n = len(self.X_train)

                y_pred = self.critic_network.predict(np.concatenate((self.X_train,self.y_train),axis=1))
                y_pred = [-1 if i < 0 else 1 for i in y_pred.ravel()]
                real_acc = accuracy_score(-np.ones(n),y_pred) * 100

                z = np.random.normal(0, 1, (n, self.rand_noise_dim))
                s_labels = np.random.choice([0,2,3,4],(n,1), replace=True)
                y_pred = self.combined.predict([z, s_labels])
                y_pred = [-1 if i < 0.0 else 1 for i in y_pred.ravel()]
                fake_acc = accuracy_score(y_pred,np.ones(n)) * 100

                print ("Epoch : {:d} [critic loss: {:.4f}, acc(real) : {:.4f}, acc(fake) : {:.4f}] [G loss: {:.4f}]".format(epoch, d_loss, real_acc, fake_acc, g_loss))

    def save_model_config():
        pass
