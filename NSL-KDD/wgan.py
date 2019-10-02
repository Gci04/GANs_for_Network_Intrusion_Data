import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras import optimizers

class WGAN(object):
    """Wasserstein Generative Adversarial Network Class"""

    def __init__(self, args):
        self.arg = arg

    def wasserstein_loss(self,y_true, y_pred):
        """define earth mover distance (wasserstein loss)"""
        return K.mean(y_true * y_pred)

    def define_generator(self):
        """Build a Generator Model"""
        return None

    def define_critic(self):
        """Build a critic""""
        return None

    def __define_models(self):
        pass

    def train(self):
        pass
