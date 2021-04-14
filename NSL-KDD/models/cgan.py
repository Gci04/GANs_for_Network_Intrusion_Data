import numpy as np
import os, pickle, tqdm
from scipy.stats import norm

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(12343)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout, Activation
from tensorflow.keras import optimizers
from collections import defaultdict
from tensorflow.keras.utils import get_custom_objects

def h_function(value):
    h_ = value
    clip_val_max = tf.math.reduce_max(tf.math.reduce_max(h_))
    h_ = tf.clip_by_value(h_,0,clip_val_max)
    h_ = pow(h_,3)*(pow(h_,5)-(2*pow(h_,4))+2)
    return h_

def h2_function(value):
    if value > 0:
        return pow(value,3)*(pow(value,5)-(2*pow(value,4))+2)
    else:
        return 0

def SPOCU_f(input):
    alpha = 7.5
    beta = 1
    gamma = 8.7
    out = alpha*h_function((input/gamma)+beta) - alpha*h2_function(beta)
    return out

class SPOCU(Activation):

    def __init__(self, activation, **kwargs):
        super(SPOCU, self).__init__(activation, **kwargs)
        self.__name__ = 'spocu'


get_custom_objects().update({'spocu': SPOCU(SPOCU_f)})
# get_custom_objects().update({'spocu': Activation(SPOCU)})

class CGAN():
    """Conditinal Generative Adversarial Network class"""
    def __init__(self,arguments,X,y):
        [self.rand_noise_dim, self.n_layers, self.tot_epochs, self.batch_size,self.D_epochs,\
         self.G_epochs, self.activation_f, self.optimizer, self.learning_rate, self.min_num_neurones] = arguments

        self.X_train = X
        self.y_train = y

        self.label_dim = y.shape[1]
        self.x_data_dim = X.shape[1]

        self.g_losses = []
        self.d_losses, self.disc_loss_real, self.disc_loss_generated = [], [], []
        self.acc_history = []
        self.kl_history = []
        self.gan_name = '_'.join(str(e) for e in arguments).replace(".","")

        d = {}

        val, count = np.unique(self.y_train.ravel(),return_counts=True)
        for v,c in zip(val,count):
            d[v] = 0.5/c

        self.sample_prob = np.array(list(map(lambda x : d.get(x),self.y_train.ravel())))
        self.sample_prob /= self.sample_prob.sum()

        # K.clear_session()
        tf.keras.backend.clear_session()
        self.__define_models()
        self.trained = False

    def build_generator(self,x,labels):
        """Create the generator model G(z,l) : z -> random noise , l -> label (condition)"""
        x = concatenate([x,labels])
        for n in range(1,self.n_layers):
            if n == 2:
                x = Dropout(0.2)(x)
            else:
                x = Dense(self.min_num_neurones*n, activation=self.activation_f)(x)
                x = Dropout(0.5)(x)

        x = Dense(self.min_num_neurones*n, activation=self.activation_f)(x)
        x = Dense(self.x_data_dim)(x)
        x = concatenate([x,labels])

        return x

    def build_discriminator(self,x):
        """Create the discrimnator model D(G(z,l)) : z -> random noise , l -> label (condition)"""
        for n in reversed(range(1,self.n_layers+1)):
            if n%2 == 0:
                # x = Dense(self.min_num_neurones*n, activation='spocu')(x)
                x = Dropout(0.2)(x)
            else:
                x = Dense(self.min_num_neurones*n, activation='relu')(x)

        x = Dense(1, activation='sigmoid')(x)

        return x

    def __define_models(self):
        """Define Generator, Discriminator & combined model"""

        # Create & Compile generator
        tf.keras.backend.clear_session()
        generator_input = Input(shape=(self.rand_noise_dim,))
        labels_tensor = Input(shape=(self.label_dim,))
        generator_output = self.build_generator(generator_input, labels_tensor)

        self.generator = Model(inputs=[generator_input, labels_tensor], outputs=[generator_output], name='generator')
        self.generator.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        K.set_value(self.generator.optimizer.lr,self.learning_rate)

        # Create & Compile generator
        discriminator_model_input = Input(shape=(self.x_data_dim + self.label_dim,))
        discriminator_output = self.build_discriminator(discriminator_model_input)

        self.discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        K.set_value(self.discriminator.optimizer.lr,self.learning_rate)

        # Build "frozen discriminator"
        frozen_discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='frozen_discriminator')
        frozen_discriminator.trainable = False

        # Debug 1/3: discriminator weights
        n_disc_trainable = len(self.discriminator.trainable_weights)

        # Debug 2/3: generator weights
        n_gen_trainable = len(self.generator.trainable_weights)

        # Build & compile combined model from frozen weights discriminator
        combined_output = frozen_discriminator(generator_output)
        self.combined = Model(inputs = [generator_input, labels_tensor],outputs = [combined_output],name='adversarial_model')
        self.combined.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        K.set_value(self.combined.optimizer.lr,self.learning_rate)

        # Debug 3/3: compare if trainable weights correct
        # assert(len(self.discriminator._collected_trainable_weights) == n_disc_trainable)
        # assert(len(self.combined._collected_trainable_weights) == n_gen_trainable)

    def __get_batch_idx(self):
        """random selects batch_size samples indeces from training data"""
        # batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=False)
        batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=True,p = self.sample_prob)

        return batch_ix

    def train(self):
        """Trains the CGAN model"""
        print("Conditional GAN Training : [STARTED]")
        # Adversarial ground truths
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        p = norm.pdf(self.X_train.T)
        self.norm_p = p/p.sum(axis=1,keepdims=1)

        for epoch in tqdm.tqdm(range(self.tot_epochs),desc='GAN Train loop'):
            #Train Discriminator
            for i in range(self.D_epochs):

                idx = self.__get_batch_idx()
                x, labels = self.X_train[idx], self.y_train[idx]

                #Sample noise as generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.rand_noise_dim))

                #Generate a half batch of new images
                generated_x = self.generator.predict([noise, labels])

                #Train the discriminator
                d_loss_fake = self.discriminator.train_on_batch(generated_x, fake_labels)
                d_loss_real = self.discriminator.train_on_batch(np.concatenate((x,labels),axis=1), real_labels)
                d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
                print("d_loss : {}".format(d_loss[0]))

            self.disc_loss_real.append(d_loss_real[0])
            self.disc_loss_generated.append(d_loss_fake[0])
            self.d_losses.append(d_loss[0])
            self.acc_history.append([d_loss_fake[1],d_loss_real[1]])

            #Train Generator (generator in combined model is trainable while discrimnator is frozen)
            for j in range(self.G_epochs):
                #Condition on labels
                sampled_labels = np.random.choice(np.unique(self.y_train),(self.batch_size,1), replace=True)

                #Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], real_labels)
                self.g_losses.append(g_loss[0])

            if epoch % 10 == 0:
                self.calculate_kl_div()

        self.trained = True
        print("Conditional GAN Training : [DONE]")

    def generate_data(self,labels):
        n = len(labels)
        assert(self.trained == True), "Model not trained!!"
        assert(n != 0 ), "Labels Empty!!"
        noise = np.random.normal(0, 1, (n, self.rand_noise_dim))
        return self.generator.predict([noise, labels])[:,:-1]

    def dump_to_file(self,save_dir = "./logs"):
        """Dumps the training history and GAN config to pickle file """
        H = defaultdict(dict)
        H["acc_history"] = self.acc_history
        H["Generator_loss"] = self.g_losses
        H["disc_loss_real"] = self.disc_loss_real
        H["disc_loss_gen"] = self.disc_loss_generated
        H["discriminator_loss"] = self.d_losses
        H["kl_divergence"] = self.kl_history
        H["rand_noise_dim"] , H["total_epochs"] = self.rand_noise_dim, self.tot_epochs
        H["batch_size"] , H["learning_rate"]  = self.batch_size, self.learning_rate
        H["n_layers"] , H["activation"]  = self.n_layers , self.activation_f
        H["optimizer"] , H["min_num_neurones"] = self.optimizer, self.min_num_neurones

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(f"{save_dir}/CGAN_{self.gan_name}{'.pickle'}", "wb") as output_file:
            pickle.dump(H,output_file)

        generator_save_dir = './trained_generator'
        if not os.path.exists(generator_save_dir):
            os.makedirs(generator_save_dir)
        self.generator.save(os.path.join(generator_save_dir,'gen.h5'))
        print("Save Model : DONE")

    def calculate_kl_div(self):
        """
        calculate Kullback–Leibler divergence between the generated dataset and original dataset.
        Source : https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
        """
        # K.set_learning_phase(0)
        self.generator.trainable = False
        noise = np.random.normal(0, 1, (len(self.X_train), self.rand_noise_dim))
        g_z = self.generator.predict([noise, self.y_train])[:,:-1]
        self.generator.trainable = True

        q = norm.pdf(g_z.T)
        norm_q = q/q.sum(axis=1,keepdims=1)

        kl = np.sum(np.where(self.norm_p != 0, self.norm_p * np.log(self.norm_p/norm_q),0))

        self.kl_history.append(kl)
