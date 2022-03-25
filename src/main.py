from data.DataGeneration import DataGeneration
from layers.Linear import Linear
from module.Sequential import Sequential
from utils.utils import *

import numpy as np
np.random.seed(42)


if __name__ == '__main__':
	data_generation = DataGeneration(points=2000, classes=2)
	data_generation.make_vertical_data()
	# data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	n_features = X.shape[1]
	n_neurons = 64 
	n_classes = len(np.unique(y))	
	learning_rate = 1e-5
	n_epochs = 5
	train_split = 0.25

	train_x, test_x, train_y, test_y = split_data(X, y, train_split=train_split, shuffle=True)

	## Gradient exploded when going for 128 neurons per hidden layer and SGD
	## Gradient exploded when going for GD
	model = Sequential()
	model.add(layer=Linear(n_features, n_neurons), activation="tanh")
	model.add(layer=Linear(n_neurons, n_classes), activation="sigmoid")
	model.compile(loss="sparse_binary_crossentropy", 
				  optimizer="SGD",
				  learning_rate=learning_rate)
	model.summary()
	model.fit(train_x, train_y, n_epochs=n_epochs)

	model.stats()
	print("--> Accuracy:", model.score(train_x, train_y, type="accuracy"))
	print("--> Accuracy:", model.score(test_x, test_y, type="accuracy"))