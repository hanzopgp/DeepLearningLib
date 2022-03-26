from data.DataGeneration import DataGeneration
from layers.Linear import Linear
from network.Sequential import Sequential
from utils.utils import *

import numpy as np
np.random.seed(42)

## TODOLIST:
## Fix gradient exploding
## Fix loss being the same for train and valid while using score

## * Activation functions:
## --> hidden_layer    : relu, sigmoid, softmax, tanh
## --> output layer    : sigmoid, softmax
## * Loss functions:
## --> classification  : binary_crossentropy, categorical_crossentropy, sparse_binary_crossentropy, sparse_categorical_crossentropy
## --> regression      : mse, mae, rmse
## * Optimizer functions:
## --> basic optimizer : GD, SGD, MGD
## * Score types:
## classification score: accuracy

if __name__ == '__main__':
	data_generation = DataGeneration(points=10_000, classes=2)
	data_generation.make_vertical_data()
	# data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	n_features = X.shape[1]
	n_neurons = 16
	n_classes = len(np.unique(y))	
	learning_rate = 1e-5
	n_epochs = 50
	train_split = 0.2
	n_batch = 10 ## In case we use MGD

	X, valid_x, y, valid_y = split_data(X, y, train_split=train_split, shuffle=True)
	train_x, test_x, train_y, test_y = split_data(X, y, train_split=train_split, shuffle=True)

	## Gradient exploded when going for 128 neurons per hidden layer and SGD
	## Gradient exploded when going for GD
	## Gradient exploded when going for 2 hidden layers
	## Going for a lower learning rate fixes the problem but I'm not sure its intended
	model = Sequential()
	model.add(layer=Linear(n_features, n_neurons), activation="tanh")
	# model.add(layer=Linear(n_neurons, n_neurons), activation="tanh")
	model.add(layer=Linear(n_neurons, n_classes), activation="sigmoid")
	model.compile(loss="sparse_binary_crossentropy", 
				  optimizer="MGD",
				  learning_rate=learning_rate,
				  metric="accuracy",
				  n_batch=n_batch)
	model.summary()
	model.fit(train_x, train_y, valid_x, valid_y, n_epochs=n_epochs, verbose=True)

	model.plot_stats()
	# _, train_acc, _, test_acc = model.compute_scores(type="accuracy")
	# print("--> Accuracy in train :", train_acc)
	# print("--> Accuracy in test :", test_acc)