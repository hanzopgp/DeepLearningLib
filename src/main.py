from data.DataGeneration import DataGeneration
from layers.Linear import Linear
from network.Sequential import Sequential
from utils.utils import *

import numpy as np
np.random.seed(42)


## ToDoList:
## Fix gradient exploding (is it a bug or we need gradient clipping)
## Enhance loss/metric computation, avoid forward pass somehow
## MSGD is not working, don't know how to implement at the moment
## Try regression
## Build test package with all the examples that works !!! (one per .py file)

## What currently works:
## Binary classification on vertical data
## Multiclass classification on vertical data
## Regression ???
## Binary classification on spiral data ???
## Multiclass classification on spiral data ???
## Binary classification on real data () ???
## Multiclass classification on real data (USPS) ???


## * Activation functions:
## --> hidden_layer    : relu, sigmoid, softmax, tanh
## --> output layer    : sigmoid, softmax

## * Loss functions:
## --> classification  : binary_crossentropy, categorical_crossentropy, sparse_binary_crossentropy, sparse_categorical_crossentropy
## if the output activation is softmax and we use sparse_categorical_crossentropy then the loss is a SparseCategoricalCrossEntropySoftmax()
## --> regression      : mse, mae, rmse

## * Optimizer functions:
## --> basic optimizer : GD, SGD, MGD, MSGD

## * Score types:
## classification score: accuracy


if __name__ == '__main__':
	data_generation = DataGeneration(points=1_000, classes=3)
	data_generation.make_vertical_data()
	# data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	n_features = X.shape[1]
	n_neurons = 64
	n_classes = len(np.unique(y))	
	learning_rate = 1e-3
	n_epochs = 50
	train_split = 0.2
	n_batch = 10 ## In case we use MGD
	gamma = 0.9  ## In case we use MSGD

	X, valid_x, y, valid_y = split_data(X, y, train_split=train_split, shuffle=True)
	train_x, test_x, train_y, test_y = split_data(X, y, train_split=train_split, shuffle=True)

	## Gradient exploded when going for 128 neurons per hidden layer and SGD
	## Gradient exploded when going for GD
	## Gradient exploded when going for 2 hidden layers
	## Going for a lower learning rate fixes the problem but I'm not sure its intended
	## Maybe we should go for gradient clipping techniques if it's not a bug
	model = Sequential()
	model.add(layer=Linear(n_features, n_neurons), activation="tanh")
	# model.add(layer=Linear(n_neurons, n_neurons), activation="tanh")
	model.add(layer=Linear(n_neurons, n_classes), activation="softmax")
	model.compile(loss="sparse_categorical_crossentropy", 
				  #loss="sparse_binary_crossentropy",
				  #loss="mse",
				  optimizer="SGD",
				  learning_rate=learning_rate,
				  metric="accuracy",
				  n_batch=n_batch, ## If we use MGD
				  gamma=gamma)     ## If we use MSGD
	model.summary()
	model.fit(train_x, 
			  train_y, 
			  valid_x, 
			  valid_y, 
			  n_epochs=n_epochs, 
			  verbose=True)
	# model.plot_stats()