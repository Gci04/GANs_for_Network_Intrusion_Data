import numpy as np
import os, pickle, tqdm
from scipy.stats import norm

import torch
from torch import nn
from spocu.spocu_pytorch import SPOCU

class Discriminator(nn.Module):

    def __init__(self, arg):
        super(Discriminator, self).__init__()
        self.arg = arg

    def foward(self, input, label):
        pass

class Generator(object):

    def __init__(self, arg):
        super(Generator, self).__init__()
        self.arg = arg
        self.model = nn.Sequential()

    def foward(self, input, label):
        pass


class CGAN():
    """Conditinal Generative Adversarial Network class"""
    def __init__(self,arguments,X,y):
        [self.rand_noise_dim, self.tot_epochs, self.batch_size, self.learning_rate] = arguments

        self.X_train = X
        self.y_train = y

        self.to_generate = torch.unique(y)
        self.label_dim = y.shape[1]
        self.x_data_dim = X.shape[1]

        self.g_losses = []
        self.d_losses, self.disc_loss_real, self.disc_loss_generated = [], [], []
        self.acc_history = []
        self.kl_history = []
        self.gan_name = '_'.join(str(e) for e in arguments).replace(".","")

        d = {}
        val, count = torch.unique(self.y_train,return_counts=True)
        for v,c in zip(val,count):
            d[v.item()] = 0.5/c.item()

        self.sample_prob = np.array(list(map(lambda x : d.get(x),self.y_train.numpy().ravel())))
        self.sample_prob /= self.sample_prob.sum()

        self.__define_models()
        self.trained = False


    def __define_models(self):
        """Define Generator, Discriminator & combined model"""

        # Create & Compile generator
        self.generator = Generator()

        # Create & Compile generator
        self.discriminator = Discriminator()

    def __get_batch_idx(self):
        """random selects batch_size samples indeces from training data"""
        if self.batch_size > 130:
            return np.random.choice(len(self.X_train), size=self.batch_size, replace=True,p = self.sample_prob)
        else:
            return np.random.choice(len(self.X_train), size=self.batch_size, replace=False,p = self.sample_prob)

    def train(self):
        """Trains the CGAN model"""
        print("Conditional GAN Training : [Started]")
        # Adversarial ground truths
        real_labels = torch.ones((self.batch_size, 1))
        fake_labels = torch.zeros((self.batch_size, 1))

        p = norm.pdf(self.X_train.numpy().T)
        self.norm_p = p/p.sum(axis=1,keepdims=1)

        for epoch in tqdm.tqdm(range(self.tot_epochs),desc='cGAN training '):
            #Train Discriminator

            idx = self.__get_batch_idx()
            x, labels = self.X_train[idx], self.y_train[idx]

                #Sample noise as generator input
            noise = torch.normal(0, 1, (self.batch_size, self.rand_noise_dim))
            #Generate a half batch of new samples
            generated_x = self.generator(noise, labels)

            d_loss = 0.5 * np.add(d_loss_real.item(),d_loss_fake.item())

            self.disc_loss_real.append(d_loss_real[0])
            self.disc_loss_generated.append(d_loss_fake[0])
            self.d_losses.append(d_loss[0])
            self.acc_history.append([d_loss_fake[1],d_loss_real[1]])

            #Train Generator (generator in combined model is trainable while discrimnator is frozen)
            #Condition on labels
            sampled_labels = np.random.choice(self.to_generate,(self.batch_size,1), replace=True)

            #Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], real_labels)
            self.g_losses.append(g_loss[0])


            #Print metrices
            # print ("Epoch : {:d} [D loss: {:.4f}, acc.: {:.4f}] [G loss: {:.4f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
        self.trained = True
        print("Conditional GAN Train : [Finished]")

    def calculate_kl_div(self):
        """
        calculate Kullback–Leibler divergence between the generated dataset and original dataset.
        Source : https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
        """
        # K.set_learning_phase(0)
        self.generator.eval()
        noise = np.random.normal(0, 1, (len(self.X_train), self.rand_noise_dim))
        g_z = self.generator(noise, self.y_train)
        # self.generator.train()

        q = norm.pdf(g_z.numpy().T)
        norm_q = q/q.sum(axis=1,keepdims=1)

        kl = np.sum(np.where(self.norm_p != 0, self.norm_p * np.log(self.norm_p/norm_q),0))
        # print(" KL : {}".format(kl))
        # K.set_learning_phase(1)

        self.kl_history.append(kl)

    def generate_data(self,labels):
        n = len(labels)
        assert(self.trained == True), "Model not trained!!"
        assert(n != 0 ), "Labels Empty!!"
        noise = torch.normal(0, 1, (n, self.rand_noise_dim))

        return self.generator(noise, labels)

    def dump_to_file(self,save_dir = "./logs"):
        """Dumps the training history and GAN config to pickle file """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pass
