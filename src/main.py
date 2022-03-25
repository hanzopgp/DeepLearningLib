from data.DataGeneration import DataGeneration
from layers.Linear import Linear
from module.Sequential import Sequential
from utils.utils import *

import numpy as np


if __name__ == '__main__':
	data_generation = DataGeneration(points=500, classes=2)
	data_generation.make_vertical_data()
	data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	n_features = X.shape[1]
	n_neurons = 128
	n_classes = len(np.unique(y))	
	learning_rate = 1e-3
	n_epochs = 10

	model = Sequential()
	model.add(layer=Linear(n_features, n_neurons), activation="tanh")
	# model.add(layer=Linear(n_neurons, n_neurons), activation="tanh")
	model.add(layer=Linear(n_neurons, n_classes), activation="sigmoid")
	model.compile(loss="sparse_binary_crossentropy", learning_rate=learning_rate)
	model.summary()
	model.fit(X, y, n_epochs=n_epochs)

	model.stats()
	# model.score(X, y, type="accuracy")