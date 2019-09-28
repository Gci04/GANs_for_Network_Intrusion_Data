import numpy as np
import pandas as pd
import os, sys, matplotlib
import catboost as cat
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input, concatenate,LeakyReLU
from keras import optimizers,engine
from scipy.stats import norm
from keras.losses import kullback_leibler_divergence

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generator_network(z, data_dim, min_num_neurones):
    x = Dense(min_num_neurones, activation='tanh')(z)
    x = Dense(min_num_neurones*2, activation='tanh')(x)
    x = Dense(min_num_neurones*4, activation='tanh')(x)
    # x = Dense(min_num_neurones*8, activation='linear')(x)
    x = Dense(data_dim,activation='linear')(x)
    return x

def discriminator_network(x, data_dim, min_num_neurones):
    x = Dense(min_num_neurones*4, activation='tanh')(x)
    x = Dense(min_num_neurones*2, activation='tanh')(x)
    x = Dense(min_num_neurones, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)
    return x

def plot_data(real,fake):
    """
    Visualization of the real and generated data to visualize the distribution differences.

    Parameters:
    ------------
    real : ndarray
        samples of the real data
    fake : ndarray
        samples of the generated data

    Return Value : None
    """
    fake_pca = PCA(n_components=2)
    fake_data = fake_pca.fit_transform(fake)

    real_pca = PCA(n_components=2)
    real_data = real_pca.fit_transform(real)

    _ , ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].scatter( real_data[:,0], real_data[:,1],color="g")
    ax[1].scatter( fake_data[:,0], fake_data[:,1],color="r")
    # ax[0].scatter( real[:,0], real[:,5],color="g")
    # ax[1].scatter( fake[:,0], fake[:,5],color="r")

    ax[0].set_title('Real')
    ax[1].set_title('Generated')

    ax[1].set_xlim(ax[0].get_xlim()), ax[1].set_ylim(ax[0].get_ylim())

    plt.show(block=False)
    plt.pause(3)
    plt.close()

def plot_distributions(real_dist,generated_dist):
    """
    Plot top and bottom 3 based on the KL-divergence value
    """

    kl_values = np.sum(np.where(real_dist != 0, real_dist * np.log(real_dist/generated_dist),0),axis=1)
    top_3 = np.argsort(kl_values)[:3]
    bottom = np.argsort(kl_values)[-3:]

    tot_features = top_3.tolist() + bottom.tolist()
    sns.set(rc={'figure.figsize':(12,8)},font_scale=1.3)
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle('Distributions of Features')

    for ax, feature, name in zip(axes.flatten(), tot_features , tot_features):
        sns.distplot(real_dist[feature], hist = False, kde = True, ax = ax,
                     kde_kws = {'shade': True, 'linewidth': 3}, label = "Real")
        sns.distplot(generated_dist[feature], hist = False, kde = True,ax = ax,
                     kde_kws = {'shade': True, 'linewidth': 3}, label = "Fake")

        ax.set(title=f'Feature : {name}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show(block=False)
    plt.pause(5)
    plt.close()

def define_models_GAN(z_dim, data_dim, min_num_neurones,learning_rate):
    """
    Put together the generator model and discriminator model (Define the full Generative Adversarial Network).

    Parameters:
    -----------
    z_dim : int default = 1
        The dimmension of random noise
    data_dim : int
        The dimmension/size of the original Data (which will be genarated/faked)
    min_num_neurones : int
        The base/min count of neurones in each nn layer

    Return Value:
    -------------
    generator_model : Model
        GAN generator Model (Keras model) G(z) -> x
    discriminator_model : Model
        Discriminator Model which will dertimine if a sample is fake or not. D(x)
    adversarial_model : model
        Generator + discriminator ==> (D(G(z)))
    """
    adam = optimizers.Adam(lr= learning_rate, beta_1=0.5, beta_2=0.9)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    sgd = optimizers.SGD(lr=learning_rate, clipvalue=0.5)

    # Create & Compile discriminator
    discriminator_model_input = Input(shape=(data_dim,))
    discriminator_output = discriminator_network(discriminator_model_input, data_dim, min_num_neurones)

    discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='discriminator')
    discriminator.compile(loss='binary_crossentropy',optimizer=sgd)

    # Build "frozen discriminator"
    frozen_discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='frozen_discriminator')
    frozen_discriminator.trainable = False
    # frozen_discriminator.compile(loss='binary_crossentropy',optimizer=adam)


    # Debug 1/3: discriminator weights
    n_disc_trainable = len(discriminator.trainable_weights)

    # Create & Compile generator
    generator_input = Input(shape=(z_dim, ))
    generator_output = generator_network(generator_input, data_dim, min_num_neurones)
    generator = Model(inputs=[generator_input], outputs=[generator_output], name='generator')
    # generator.compile(loss='binary_crossentropy',optimizer=adam)

    # Debug 2/3: generator weights
    n_gen_trainable = len(generator.trainable_weights)

    # Build & compile adversarial model
    combined_output = frozen_discriminator(generator_output)
    # combined_output = frozen_discriminator(generator(generator_input))
    adversarial_model = Model(inputs = [generator_input],outputs = [combined_output],name='adversarial_model')
    adversarial_model.compile(loss='binary_crossentropy',optimizer=sgd)

    # Debug 3/3: compare if trainable weights correct
    assert(len(discriminator._collected_trainable_weights) == n_disc_trainable)
    assert(len(adversarial_model._collected_trainable_weights) == n_gen_trainable)

    return generator,discriminator,adversarial_model

def get_batch(X, batch_size=1):
    """
    Parameters:
    -----------
    X : ndarray
        The input data to sample a into batch
    size : int (default = 1)
        Batch size

    Return Value: ndarray - random choice of samples from the input X of batch_size
    """
    batch_ix = np.random.choice(len(X), batch_size, replace=False)
    return X[batch_ix]


def classifierAccuracy(X,g_z,n_samples):
    """
    Check how good are the generated samples using a catboost classifier. 50% accuracy
    implies that the real samples are identical and the classifier can not differentiate.
    The classifier is trained with 0.5 n_real + 1/2 n_fake samples.

    Parameters:
    -----------
    X : ndarray
        The real samples
    g_z : ndarray
        The fake samples produced by a generator

    Rerurn Value:
    -------------
    accuracy : float32
    """
    y_fake = np.ones(n_samples)
    y_true = np.zeros(n_samples)

    n = n_samples//2
    x_train = np.vstack((X[:n,:],g_z[:n,:]))
    y_train = np.hstack((np.zeros(n),np.ones(n))) # fake : 1, real : 0

    x_test = np.vstack((X[n:,:],g_z[n:,:]))
    y_test = np.hstack((np.zeros(n),np.ones(n_samples - n)))

    catb = cat.CatBoostClassifier(verbose=0,n_estimators=13,max_depth=5)
    catb.fit(x_train,y_train)

    pred = catb.predict(x_test)
    accuracy = accuracy_score(y_test,pred)

    return np.round(accuracy,decimals=3)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p/q),0))

def modelAccuracy(gen_pred,real_pred):
    gen_pred = np.array([1.0 if i > 0.5 else 0.0 for i in gen_pred])
    gen_true = np.zeros(len(gen_pred))

    real_pred = np.array([1.0 if i > 0.5 else 0.0 for i in real_pred])
    real_true = np.ones(len(gen_pred))

    print('Discriminator accuracy on Fake : {}, Real : {}'.format(accuracy_score(gen_pred,gen_true),accuracy_score(real_pred,real_true)))


def training_steps(model_components):

    [train, data_dim,generator_model, discriminator_model, combined_model,rand_dim, nb_steps,
    batch_size, D_epochs, G_epochs,combined_loss, disc_loss_generated, disc_loss_real] = model_components

    matplotlib.use('TkAgg')
    for i in range(nb_steps):
        K.set_learning_phase(1)

        # train the discriminator
        for j in range(D_epochs):
            np.random.seed(i+j)
            z = np.random.normal(size=(batch_size, rand_dim))
            x = get_batch(train, batch_size)

            fake = generator_model.predict(z)

            #get discriminator loss on real samples (d_l_r) and Generated/faked (d_l_g)
            d_l_g = discriminator_model.train_on_batch(fake, np.random.uniform(low=0.0, high=0.0001, size=batch_size))
            d_l_r = discriminator_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size))


        disc_loss_real.append(d_l_r)
        disc_loss_generated.append(d_l_g)

        #train generator
        loss = None
        for j in range(G_epochs):
            # np.random.seed(i+j)
            # z = np.random.normal(size=(batch_size, rand_dim))

            loss = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size))

        combined_loss.append(loss)

        # Checkpoint : get the lossess summery (for Generator and discriminator)
        if i % 10 == 0:
            K.set_learning_phase(0)
            test_size = len(train)

            z = np.random.normal(size=(test_size, rand_dim))
            g_z = generator_model.predict(z)

            # acc = classifierAccuracy(train,g_z,test_size)

            p = norm.pdf(train.T)
            q = norm.pdf(g_z.T)

            #https://mathoverflow.net/questions/43849/how-to-ensure-the-non-negativity-of-kullback-leibler-divergence-kld-metric-rela
            norm_p = p/p.sum(axis=1,keepdims=1)
            norm_q = q/q.sum(axis=1,keepdims=1)
            #
            # kl = kl_divergence(norm_p,norm_q)
            # tf_kl = kullback_leibler_divergence(tf.convert_to_tensor(norm_p, np.float32), tf.convert_to_tensor(norm_q, np.float32))
            # with tf.Session() as sess:
            #     print("Tensorflow kullback_leibler_divergence : {}".format(round(sum(sess.run(tf_kl)))))
            #
            # print("Ephoc : {} , Generator loss : {} , KL : {}".format(i,loss,kl))
            # print("Loss on fake: {}, Loss on real : {}".format(d_l_g, d_l_r))

            fake_pred = np.array(combined_model.predict(z)).ravel()
            real_pred = np.array(discriminator_model.predict(train)).ravel()

            modelAccuracy(fake_pred,real_pred)

            # plot_data(train,g_z)
            # plot_data(train,g_z)
            plot_distributions(norm_p,norm_q)

    return [combined_loss, disc_loss_generated, disc_loss_real]

def adversarial_training_GAN(arguments, train):

    [rand_noise_dim, nb_steps, batch_size,D_epochs, G_epochs, learning_rate, min_num_neurones] = arguments

    data_dim = train.shape[1]

    # define network models
    K.set_learning_phase(1) # 1 = train

    generator_model, discriminator_model, combined_model = define_models_GAN(rand_noise_dim, data_dim, min_num_neurones,learning_rate)

    print(generator_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    combined_loss, disc_loss_generated, disc_loss_real = [], [], []

    model_components = [train, data_dim,generator_model, discriminator_model, combined_model,
                        rand_noise_dim, nb_steps, batch_size, D_epochs, G_epochs,
                        combined_loss, disc_loss_generated, disc_loss_real ]

    [combined_loss, disc_loss_generated, disc_loss_real] = training_steps(model_components)

    discriminator_loss = (np.array(disc_loss_real) + np.array(disc_loss_generated))/2.0

    return dict({"generator_model":generator_model,"discriminator_model":discriminator_model,\
            "combined_model":combined_model,"generator_loss":combined_loss,\
            "disc_loss_generated":disc_loss_generated,"disc_loss_real": disc_loss_real,\
            "discriminator_loss":discriminator_loss})
