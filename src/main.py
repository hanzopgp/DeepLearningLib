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
	sigmoid = Sigmoid()
	tanh = Tanh()
	bce = BinaryCrossEntropy()

	## Forward pass
	model = Sequential()
	model.add(linear1)
	model.add(tanh)
	model.add(linear2)
	model.add(sigmoid)
	model.add(bce)
	model.forward(X)

	## Backward pass
	# res_bce = bce.forward(y.reshape(-1,1), res_lin)
	# delta_mse = bce.backward(y.reshape(-1,1), res_sig)
	# gradient = linear.backward_update_gradient(X, delta_mse)
	# delta_lin = linear.backward_delta(X, delta_mse)

	## Updating parameters
	# linear.update_parameters()

	## 2nd Forward pass
	# linear.zero_grad()
	# res_lin = linear.forward(X)

	## 2nd Backward pass
	# res_bce = bce.forward(y.reshape(-1,1), res_lin)
	# delta_mse = bce.backward(y.reshape(-1,1), res_lin)
	# gradient = linear.backward_update_gradient(X, delta_mse)
	# delta_lin = linear.backward_delta(X, delta_mse)
	