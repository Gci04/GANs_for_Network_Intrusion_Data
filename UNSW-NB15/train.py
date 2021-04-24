import numpy as np
import pandas as pd
import os, torch

from utils import preprocessing
from utils import utils

from models import cgan
import models.classifiers as clfrs

def main(arguments):
    #Load data & preprocess
    pass

if __name__ == '__main__':
    gan_params = [32, 4, 2000, 128, 1, 1, 'relu', 'sgd', 0.0005, 27]
    main(gan_params)
