from data.DataGeneration import DataGeneration

from layers.Linear import Linear

from activation_functions.ReLU import ReLU
from activation_functions.Sigmoid import Sigmoid
from activation_functions.Softmax import Softmax
from activation_functions.Tanh import Tanh

from loss_functions.MSE import MSE
from loss_functions.BinaryCrossEntropy import BinaryCrossEntropy
from loss_functions.CrossEntropy import CrossEntropy
from loss_functions.SparseBinaryCrossEntropy import SparseBinaryCrossEntropy
from loss_functions.SparseCrossEntropy import SparseCrossEntropy

from module.Sequential import Sequential

from utils.utils import *

import numpy as np


if __name__ == '__main__':
	data_generation = DataGeneration(points=100, classes=2)
	data_generation.make_vertical_data()
	# data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	linear1 = Linear(X.shape[1], 32)
	linear2 = Linear(32, len(np.unique(y)))
	sbce = SparseBinaryCrossEntropy()
	learning_rate = 1e-3
	n_epochs = 10

	model = Sequential()
	model.add(layer=linear1, activation="tanh")
	model.add(layer=linear2, activation="sigmoid")
	model.compile(loss_function=sbce, learning_rate=learning_rate)
	model.summary()
	model.fit(X, y, n_epochs=n_epochs)

	# pred = model.predict(X)
	# print(model.score(pred, y))