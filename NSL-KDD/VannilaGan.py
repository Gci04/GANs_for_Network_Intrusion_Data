import numpy as np
import matplotlib, pickle, os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from collections import defaultdict

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras import optimizers

import utils

class Vannila_GAN():
    """Vannila Generative Adversarial Network Class"""

    def __init__(self,arguments,X):
        [self.rand_noise_dim, self.tot_epochs, self.batch_size,self.D_epochs,\
         self.G_epochs, self.activation_f, self.optimizer, self.learning_rate, self.min_num_neurones] = arguments

        self.X_train = X
        self.x_data_dim = X.shape[1]

        self.sgd = optimizers.SGD(lr=self.learning_rate)

        self.disc_loss_real, self.disc_loss_generated , self.discriminator_loss= [], [], []
        self.generator, self.discriminator, self.combined_model = None, None, None
        self.combined_loss = []

        self.kl_history = []
        self.acc_history = []

        self.__define_models()

    def generator_network(self,x):
        """generator model definition"""
        x = Dense(self.min_num_neurones, activation=self.activation_f)(x)
        x = Dense(self.min_num_neurones*2, activation=self.activation_f)(x)
        x = Dense(self.min_num_neurones*4, activation=self.activation_f)(x)
        x = Dense(self.x_data_dim,activation='linear')(x)

        return x

    def discriminator_network(self,x):
        """discriminator model definition"""
        x = Dense(self.min_num_neurones*4, activation=self.activation_f)(x)
        x = Dense(self.min_num_neurones*2, activation=self.activation_f)(x)
        x = Dense(self.min_num_neurones, activation=self.activation_f)(x)
        x = Dense(1, activation='sigmoid')(x)

        return x

    def __get_batch_idx(self):
        """random selects batch_size samples indeces from training data"""
        batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=False)

        return batch_ix

    def __define_models(self):
        """Define Generator, Discriminator & combined model"""
        # Create & Compile discriminator
        discriminator_model_input = Input(shape=(self.x_data_dim,))
        discriminator_output = self.discriminator_network(discriminator_model_input)

        self.discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy',optimizer=self.optimizer)
        K.set_value(self.discriminator.optimizer.lr,self.learning_rate)

        # Build "frozen discriminator"
        frozen_discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='frozen_discriminator')
        frozen_discriminator.trainable = False
        # frozen_discriminator.compile(loss='binary_crossentropy',optimizer=adam)

        # Debug 1/3: discriminator weights
        n_disc_trainable = len(self.discriminator.trainable_weights)

        # Create & Compile generator
        generator_input = Input(shape=(self.rand_noise_dim, ))
        generator_output = self.generator_network(generator_input)
        self.generator = Model(inputs=[generator_input], outputs=[generator_output], name='generator')
        self.generator.compile(loss='binary_crossentropy',optimizer=self.optimizer)
        K.set_value(self.generator.optimizer.lr,self.learning_rate)
        # Debug 2/3: generator weights
        n_gen_trainable = len(self.generator.trainable_weights)

        # Build & compile adversarial model (Generator + Discriminator)
        combined_output = frozen_discriminator(generator_output)

        self.combined_model = Model(inputs = [generator_input],outputs = [combined_output],name='combined_model')
        self.combined_model.compile(loss='binary_crossentropy',optimizer=self.optimizer)
        K.set_value(self.combined_model.optimizer.lr,self.learning_rate)

        # Debug 3/3: compare if trainable weights correct
        assert(len(self.discriminator._collected_trainable_weights) == n_disc_trainable)
        assert(len(self.combined_model._collected_trainable_weights) == n_gen_trainable)

    def train(self,plot_dist = False):
        matplotlib.use('TkAgg')
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        p = utils.norm.pdf(self.X_train.T)
        norm_p = p/p.sum(axis=1,keepdims=1)

        for i in range(self.tot_epochs):
            K.set_learning_phase(1)
            # train the discriminator
            for j in range(self.D_epochs):
                np.random.seed(i+j)
                z = np.random.normal(size=(self.batch_size, self.rand_noise_dim))
                idx = self.__get_batch_idx()
                x = self.X_train[idx]
                fake = self.generator.predict(z)

                #get discriminator loss on real samples (d_l_r) and Generated/faked (d_l_g)
                # d_l_g = self.discriminator.train_on_batch(fake, np.random.uniform(low=0.0, high=0.0001, size=self.batch_size))
                # d_l_r = self.discriminator.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=self.batch_size))

                d_l_g = self.discriminator.train_on_batch(fake, fake_labels)
                d_l_r = self.discriminator.train_on_batch(x, real_labels)
                d_loss = 0.5 * np.add(d_l_r,d_l_g)

            self.disc_loss_real.append(d_l_r)
            self.disc_loss_generated.append(d_l_g)
            self.discriminator_loss.append(d_loss)

            #train generator
            loss = None
            for j in range(self.G_epochs):
                # np.random.seed(i+j)
                # z = np.random.normal(size=(batch_size, rand_dim))

                g_loss = self.combined_model.train_on_batch(z, real_labels)
                # loss = self.combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=self.batch_size))

            self.combined_loss.append(g_loss)

            # Checkpoint
            if i % 10 == 0:
                K.set_learning_phase(0)

                z = np.random.normal(size=(len(self.X_train), self.rand_noise_dim))
                g_z = self.generator.predict(z)

                fake_pred = np.array(self.combined_model.predict(z)).ravel()
                real_pred = np.array(self.discriminator.predict(self.X_train)).ravel()

                acc_g, acc_r = utils.modelAccuracy(fake_pred,real_pred)
                self.acc_history.append([acc_g, acc_r])

                q = utils.norm.pdf(g_z.T)
                norm_q = q/q.sum(axis=1,keepdims=1)

                if plot_dist :
                    utils.plot_distributions(norm_p,norm_q)

                kl = np.sum(np.where(norm_p != 0, norm_p * np.log(norm_p/norm_q),0))

                print("Epoch : {:d} [D loss : {:.4f}, acc_fake : {:.4f}, acc_real : {:.4f}] [G loss : {:.4f}], KL : {:.4f}".format(i, d_loss,acc_g ,acc_r, g_loss,kl))

        self.acc_history = np.array(self.acc_history)

    def save_model_componets(self,dir='vannilaGanLogs'):
        """Dumps the training history to pickle file and GAN to .h5 file """
        H = defaultdict(dict)
        H["acc_history"] = self.acc_history.tolist()
        H["G_loss"] = self.combined_loss
        H["disc_loss_real"] = self.disc_loss_real
        H["disc_loss_gen"] = self.disc_loss_generated
        H["disc_loss"] = self.discriminator_loss

        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir+"/gan_history.pickle", "wb") as output_file:
            pickle.dump(H,output_file)

        self.combined_model.save(dir+"/gan_combined_model.h5")

    def plot_model_losses(self):
        """Plot the the models losses"""
        x = np.arange(self.tot_epochs)
        plt.plot(x,self.combined_loss,label="Generator Loss")
        plt.plot(x,self.discriminator_loss,label="Discriminator Loss")
        plt.title("NLS-KDD99 GAN Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    def plot_model_acc_history(self):
        x = np.arange(0,100,10)
        plt.plot(x,self.acc_history[:,0],label="Acc D(G(z))")
        plt.plot(x,self.self.acc_history[:,1],label="Acc D(x)")
        plt.title("Discriminator accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch #")
        plt.legend()
        plt.show()
