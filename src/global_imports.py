import numpy as np
# np.random.seed(42)
from matplotlib import pyplot as plt
# import tensorflow as tf
from tqdm import tqdm

import global_variables

from data.DataGeneration import DataGeneration, ContinuousGen, MultiClassGen, TwoClassGen

from activation_functions.ReLU import ReLU
from activation_functions.Sigmoid import Sigmoid
from activation_functions.Softmax import Softmax
from activation_functions.Tanh import Tanh
from activation_functions.Lin import Lin
from activation_functions.LeakyReLU import LeakyReLU

from loss_functions.MeanSquaredError import MeanSquaredError
from loss_functions.MeanAbsoluteError import MeanAbsoluteError
from loss_functions.RootMeanSquaredError import RootMeanSquaredError
from loss_functions.BinaryCrossEntropy import BinaryCrossEntropy
from loss_functions.CategoricalCrossEntropy import CategoricalCrossEntropy
from loss_functions.SparseBinaryCrossEntropy import SparseBinaryCrossEntropy
from loss_functions.SparseCategoricalCrossEntropy import SparseCategoricalCrossEntropy
from loss_functions.SparseCategoricalCrossEntropySoftmax import SparseCategoricalCrossEntropySoftmax

from optimizer_functions.GradientDescent import GradientDescent
from optimizer_functions.StochasticGradientDescent import StochasticGradientDescent
from optimizer_functions.MinibatchGradientDescent import MinibatchGradientDescent

from layers.Linear import Linear
from network.Sequential import Sequential
from cnn.Conv import Convo1D
from cnn.Pooling import MaxPool1D, Flatten
