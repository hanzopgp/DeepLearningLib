from global_imports import *
from utils.utils import *


###### ToDoList:

## Debug cases such as nn failing to converge in some situations
## Enhance loss/metric computation, avoid forward pass somehow
## AutoEncoders (bonus : sharing weight matrixes)
## Clean early stopping code --> implement it in GD / MGD
## CNN


###### AutoEncoders ideas: (need to implement at least one and study it)

## Image reconstruction with AutoEncoders ???
## Represent 2D/3D latent space (bonus : MD latent space with T-sne) ???
## Using AutoEncoders on noisy data and observing errors ???
## Classification using latent space given by AutoEncoders ???


##### Implemented: 

## * Initialization :
## --> Xavier initialization

## * Layers :
## --> Linear()

## * Activation functions:
## --> hidden_layer    : relu, lrelu, sigmoid, softmax, tanh
## --> output layer    : sigmoid, softmax, linear

## * Loss functions:
## --> classification  : binary_crossentropy, categorical_crossentropy, sparse_binary_crossentropy, sparse_categorical_crossentropy
## if the output activation is softmax and we use sparse_categorical_crossentropy then the loss is a SparseCategoricalCrossEntropySoftmax()
## --> regression      : mse, mae, rmse

## * Optimizer functions:
## --> basic optimizer : gd, sgd, mgd

## * Score types:
## --> classification : accuracy

## * Regularization:
## --> L2 with regularization_lambda

## * Early stopping:
## --> Dictionnary with "metric", "patience", "min_delta"


if __name__ == '__main__':
	data_generation = DataGeneration(points=200, classes=3)
	data_generation.make_spiral_data()
	# data_generation.display_data()
	X, y = data_generation.x, data_generation.y

	n_features = X.shape[1]
	n_neurons = 32
	n_classes = len(np.unique(y))	
	learning_rate = 1e-3
	n_epochs = 500
	train_split = 0.2
	n_batch = 10 ## In case we use MGD
	decay = 1e-6
	init_type = "xavier"

	X, valid_x, y, valid_y = split_data(X, y, train_split=train_split, shuffle=True)
	train_x, test_x, train_y, test_y = split_data(X, y, train_split=train_split, shuffle=True)

	model = Sequential()
	model.add(layer=Linear(n_features, n_neurons, init_type=init_type), activation="tanh")
	model.add(layer=Linear(n_neurons, n_neurons, init_type=init_type), activation="tanh")
	model.add(layer=Linear(n_neurons, n_neurons, init_type=init_type), activation="tanh")
	model.add(layer=Linear(n_neurons, n_classes, init_type=init_type), activation="softmax")
	model.compile(loss="sparse_categorical_crossentropy", 
				  #loss="sparse_binary_crossentropy",
				  #loss="mse",
				  optimizer="sgd",
				  learning_rate=learning_rate,
				  metric="accuracy",
				  n_batch=n_batch, ## If we use MGD
				  decay=decay)
	model.summary()
	model.fit(train_x, 
			  train_y, 
			  valid_x, 
			  valid_y, 
			  n_epochs=n_epochs, 
			  verbose=True)
	model.plot_stats()