import numpy as np
import os, pickle, tqdm, torch
from scipy.stats import norm
from torch import nn

class Generator(nn.Module):
    # initializers
    def __init__(self,output_dim, noise_dim=32):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim+1, 27),
            nn.ReLU(),
            nn.Linear(27, 81),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(81, 108),
            nn.ReLU(),
            nn.Linear(108, output_dim),
        )

    # forward method
    def forward(self, input, label):

        x = torch.cat([input, label], 1)
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self,input_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size+1, 108),
            nn.ReLU(),
            nn.Linear(108, 81),
            nn.ReLU(),
            nn.Linear(81, 54),
            nn.ReLU(),
            nn.Linear(54, 27),
            nn.ReLU(),
            nn.Linear(27, 1),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, input, label):
        x = torch.cat([input,label], 1)
        x = self.model(x)
        return x

class CGAN():
    """Conditinal Generative Adversarial Network class"""
    def __init__(self,arguments,X,y):
        [self.rand_noise_dim, self.tot_epochs, self.batch_size, self.learning_rate] = arguments

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

        val, count = torch.unique(self.y_train,return_counts=True)
        for v,c in zip(val,count):
            d[v.item()] = 0.5/c.item()

        self.sample_prob = np.array(list(map(lambda x : d.get(x),self.y_train.numpy().ravel())))
        self.sample_prob /= self.sample_prob.sum()


        self.__define_models()
        self.trained = False


    def __define_models(self):
        """Define Generator, Discriminator & combined model"""
        self.generator = Generator(output_dim=self.x_data_dim, noise_dim=self.rand_noise_dim)
        self.discrimnator = Discriminator(input_size = self.x_data_dim)

        self.optimizer_G = torch.optim.SGD(self.generator.parameters(), lr=0.0005)
        self.optimizer_D = torch.optim.SGD(self.discrimnator.parameters(), lr=0.0005)

        self.adversarial_loss = torch.nn.BCELoss()

    def __get_batch_idx(self):
        """random selects batch_size samples indeces from training data"""
        batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=True,p = self.sample_prob)

        return batch_ix

    def train(self):
        """Trains the CGAN model"""
        print("Conditional GAN Training : [STARTED]")
        # Adversarial ground truths
        real_labels = torch.ones((self.batch_size, 1))
        fake_labels = torch.zeros((self.batch_size, 1))

        p = norm.pdf(self.X_train.T)
        self.norm_p = p/p.sum(axis=1,keepdims=1)

        # for epoch in tqdm.tqdm(range(self.tot_epochs),desc='GAN Train loop'):
        for epoch in range(self.tot_epochs):

            #Train Discriminator
            idx = self.__get_batch_idx()
            x, labels = (self.X_train[idx]), self.y_train[idx]

            #Sample noise as generator input
            noise = torch.normal(0, 1, (self.batch_size, self.rand_noise_dim))

            #  Train Discriminator
            self.optimizer_D.zero_grad()
            #Generate a half batch of new images
            d_real_loss = self.adversarial_loss(self.discrimnator(x,labels),real_labels)
            # d_real_loss.backward()
            # Loss for fake

            # with torch.no_grad():
            generated_x = self.generator(noise, labels)
            d_fake_loss = self.adversarial_loss(self.discrimnator(x,labels),fake_labels)

            d_loss = (d_real_loss + d_fake_loss) * 0.5

            d_loss.backward()
            self.optimizer_D.step()

            #Train the Generator
            self.optimizer_G.zero_grad()
            g_loss = self.adversarial_loss(self.discrimnator(generated_x,labels), real_labels)
            g_loss.backward()
            self.optimizer_G.step()

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, D_loss : {0.5 * d_loss.item() }, G_loss :  {g_loss.item()}")

        self.trained = True
        print("Conditional GAN Training : [DONE]")

    def generate_data(self,labels):
        n = len(labels)
        assert(self.trained == True), "Model not trained!!"
        assert(n != 0 ), "Labels Empty!!"
        noise = torch.normal(0, 1, (n, self.rand_noise_dim))
        self.generator.eval()
        with torch.no_grad():
            return self.generator(noise,labels)


    def dump_to_file(self,save_dir = "./logs"):
        """Dumps the training history and GAN config to pickle file """
        print("Save Model : DONE")

    def calculate_kl_div(self):
        """
        calculate Kullback–Leibler divergence between the generated dataset and original dataset.
        Source : https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
        """
        pass
