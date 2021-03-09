import tensorflow as tf  # TF 2.0
import numpy as np
import time, datetime, os, pickle
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.stats import norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, concatenate
from tensorflow.keras.utils import get_custom_objects


import pandas as pd
from utils import preprocessing
from utils import utils
from models import classifiers as clf

def plot_kl(kl_history):
    n = np.arange(0,len(kl_history),1)
    print(f"kl_divergence len {len(kl_history)}")

    fig, ax = plt.subplots()
    ax.plot(n, kl_history,label='KL',linewidth=2,color="black")
    # ax.legend(loc=0,prop={'size': 10})
    ax.set_ylabel('KL-Divergence',fontsize=11.0)
    ax.set_xlabel('Epoch',fontsize=11.0)
    ax.tick_params(labelsize=11)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('./imgs/nsl_relu_kl_div_plot.png', dpi = 350, bbox_inches='tight')
    fig.savefig('./imgs/nsl_relu_kl_div_plot.pdf', bbox_inches='tight')
    fig.savefig('./imgs/nsl_relu_kl_div_plot.eps',format='eps')

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
    alpha = 3.0937
    beta = 0.6653
    gamma = 4.437
    out = alpha*h_function((input/gamma)+beta) - alpha*h2_function(beta)
    return out

class SPOCU(Activation):

    def __init__(self, activation, **kwargs):
        super(SPOCU, self).__init__(activation, **kwargs)
        self.__name__ = 'spocu'


get_custom_objects().update({'spocu': SPOCU(SPOCU_f)})

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(128, activation='spocu')
        self.dense_2 = tf.keras.layers.Dense(256, activation='spocu')
        self.dense_3 = tf.keras.layers.Dense(512, activation='spocu')
        self.dense_4 = tf.keras.layers.Dense(25, activation='spocu')

    def call(self, noise, label):
        x = self.dense_1(tf.concat([noise, label], axis=-1))
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(512, activation='spocu')
        self.dense_2 = tf.keras.layers.Dense(256, activation='spocu')
        self.dense_3 = tf.keras.layers.Dense(128, activation='spocu')
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, inputs, label):
        x = self.dense_1(tf.concat([inputs, label], axis=-1))
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense_4(x)

def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)

def calculate_kl_div(generator_model,noise,y_labels,norm_p):
    """
    calculate Kullback–Leibler divergence between the generated dataset and original dataset.
    Source : https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
    """
    g_z = generator_model(noise, y_labels)

    q = norm.pdf(g_z.numpy().T)
    norm_q = q/q.sum(axis=1,keepdims=1)

    kl = np.sum(np.where(norm_p != 0, norm_p * np.log(norm_p/norm_q),0))

    return kl

def main(arg):
    # settting hyperparameter
    latent_dim = 32
    epochs = 100
    batch_size = 128

    #-------------------- Load Data & Preprocess ---------------------#
    train,test, label_mapping = preprocessing.get_data(encoding="Label")
    data_cols = list(train.columns[ train.columns != 'label' ])

    #Remove contant values with a threshold
    to_drop = preprocessing.get_contant_featues(train,data_cols,threshold=0.995)

    train.drop(to_drop, axis=1,inplace=True)
    test.drop(to_drop, axis=1,inplace=True)

    #Normalize row-wise the data (to unit norm) and scale column-wise
    data_cols = list(train.columns[train.columns != 'label' ])
    train = preprocessing.normalize_data(train,data_cols)
    test = preprocessing.normalize_data(test,data_cols)
    x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",True)
    data_cols = list(x_train.columns[x_train.columns != 'label' ])

    train, test = None, None
    y_train = x_train.label.values
    y_test = x_test.label.values.reshape(-1,1).astype(np.float32)

    att_ind = np.where(x_train.label != label_mapping["normal"])[0]
    for_test = np.where(x_test.label != label_mapping["normal"])[0]

    del label_mapping["normal"]
    clf.DISPLAY_PERFOMANCE = False

    x = x_train[data_cols].values[att_ind]
    y = y_train[att_ind]
    y = y.reshape(-1,1).astype(np.float32)
    x_train, y_train = None, None
    p = norm.pdf(x.T)
    norm_p = p/p.sum(axis=1,keepdims=1)

    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x),y)).batch(batch_size, drop_remainder=True)

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.SGD(0.0005)
    disc_optimizer = tf.keras.optimizers.SGD(0.0005)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    G_train_loss = tf.keras.metrics.Mean('G_train_loss', dtype=tf.float32)
    D_train_loss = tf.keras.metrics.Mean('D_train_loss', dtype=tf.float32)

    @tf.function
    def train_step(X,y_label):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_samples = generator(noise, y_label)

            real_output = discriminator(X,y_label)
            generated_output = discriminator(generated_samples, y_label)

            gen_loss = generator_loss(cross_entropy, generated_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)
        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)

        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))
        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

        return gen_loss, disc_loss

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,discriminator_optimizer=disc_optimizer,
                                    generator=generator,discriminator=discriminator)

    seed = tf.random.normal([16, latent_dim])
    kl_history = []
    g_losses = []
    disc_losses = []

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for batch_x, batch_y in train_dataset:
            gen_loss, disc_loss = train_step(batch_x,batch_y)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        # if epoch % 10 == 0 :
        noise = tf.random.normal([len(x), latent_dim])
        kl_div = calculate_kl_div(generator,noise,y.astype(np.float32),norm_p)
        kl_history.append(kl_div)
        g_losses.append(total_gen_loss / batch_size)
        disc_losses.append(total_disc_loss / batch_size)
        print('Time for epoch {:01.0f} is {:.5} sec - gen_loss = {:.5}, disc_loss = {:.5} kl_div = {:.5}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size, kl_div))

        G_train_loss.reset_states()
        D_train_loss.reset_states()

    checkpoint.save(checkpoint_prefix)
    print("Trainng finished ....")
    plot_kl(kl_history)
    print("KL plot finished ....")

    H = defaultdict(dict)
    H["Generator_loss"] = g_losses
    H["discriminator_loss"] = disc_losses
    H["kl_divergence"] = kl_history

    with open("./logs/CGAN_RELU.pickle", "wb") as output_file:
        pickle.dump(H,output_file)

    #Load Ml classifiers
    ml_classifiers = utils.load_pretrained_classifiers()
    print('pretrained ml classifiers loaded : [OK]')

    clf.compare(x,y.astype(np.float32), x_test[data_cols].values[for_test], y_test[for_test], generator, label_mapping, ml_classifiers ,cv=5,rand_noise_dim=latent_dim)

if __name__ == '__main__':
    main(None)
