from data.DataGeneration import DataGeneration

from layers.Linear import Linear

from activation_functions.ReLU import ReLU
from activation_functions.Sigmoid import Sigmoid
from activation_functions.Softmax import Softmax
from activation_functions.Tanh import Tanh

from loss_functions.MSE import MSE
from loss_functions.BinaryCrossEntropy import BinaryCrossEntropy
from loss_functions.CrossEntropy import CrossEntropy

from module.Sequential import Sequential

import numpy as np


if __name__ == '__main__':

	## Generation of some data (x1,x2 points with class y)
	data_generation = DataGeneration(points=100, classes=2)
	data_generation.make_vertical_data()
	# data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	## Hidden layers, activation functions and loss function
	linear1 = Linear(X.shape[1], 32)
	linear2 = Linear(32, len(np.unique(y)))
	bce = BinaryCrossEntropy()

	## Forward pass
	model = Sequential()
	model.add(linear1, activation="tanh")
	model.add(linear2, activation="sigmoid")
	model.compile(bce)
	model.forward(X)
	