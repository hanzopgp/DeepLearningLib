from Core import *

from activation_functions.ReLU import ReLU
from activation_functions.Sigmoid import Sigmoid
from activation_functions.Softmax import Softmax
from activation_functions.Tanh import Tanh

from loss_functions.MeanSquaredError import MeanSquaredError
from loss_functions.MeanAbsoluteError import MeanAbsoluteError
from loss_functions.RootMeanSquaredError import RootMeanSquaredError
from loss_functions.BinaryCrossEntropy import BinaryCrossEntropy
from loss_functions.CrossEntropy import CrossEntropy
from loss_functions.SparseBinaryCrossEntropy import SparseBinaryCrossEntropy
from loss_functions.SparseCrossEntropy import SparseCrossEntropy

from optimizer_functions.GradientDescent import GradientDescent
from optimizer_functions.StochasticGradientDescent import StochasticGradientDescent
from optimizer_functions.MinibatchGradientDescent import MinibatchGradientDescent


class Sequential(Module):
	def __init__(self):
		self.network = []
		self.loss_values = []
		self.acc_values = []
		self._valid = False

	def add(self, layer, activation):
		self.network.append(layer)
		if activation == "relu":
			self.network.append(ReLU())
		elif activation == "sigmoid":
			self.network.append(Sigmoid())
		elif activation == "tanh":
			self.network.append(Tanh())
		elif activation == "softmax":
			self.network.append(Softmax())
		else:
			print("Error : wrong activation function")

	def compile(self, loss, optimizer, learning_rate):
		## Choosing a loss function for our network
		if loss == "binary_crossentropy":
			loss_function = BinaryCrossEntropy()
		elif loss == "crossentropy":
			loss_function = CrossEntropy()
		elif loss == "mse":
			loss_function = MeanSquaredError()
		elif loss == "mae":
			loss_function = MeanAbsoluteError()
		elif loss == "rmse":
			loss_function = RootMeanSquaredError()
		elif loss == "sparse_binary_crossentropy":
			loss_function = SparseBinaryCrossEntropy()
		elif loss == "sparse_crossentropy":
			loss_function = SparseCrossEntropy()
		else:
			print("Error : wrong loss function")
		## Choosing and optimizer function for our network
		self._optimizer_name = optimizer
		if optimizer == "GD":
			self.optimizer = GradientDescent(self, loss_function, learning_rate)
		elif optimizer == "SGD":
			self.optimizer = StochasticGradientDescent(self, loss_function, learning_rate)
		elif optimizer == "MGD":
			self.optimizer = MinibatchGradientDescent(self, loss_function, learning_rate)
		else:
			print("Error : wrong optimizer")

	def fit(self, *arg, n_epochs, verbose):
		## If there is just a train set
		if len(arg) == 2:
			self._X = arg[0]
			self._y = arg[1]
		## If there is a train set and a valid set
		elif len(arg) == 4:
			self._X = arg[0]
			self._y = arg[1]
			self._valid_x = arg[2]
			self._valid_y = arg[3]
			self._valid = True
		else:
			print("Error : number of arguments in fit()")
		for cpt_epoch in range(n_epochs):
			self.optimizer.step(self._X, self._y)
			self.update_stats()
			if verbose == True: 
				self.show_updates(cpt_epoch=cpt_epoch)
			
	def predict(self, X):
		last_module = len(self.network) - 1
		self.network[0].forward(X)
		for i in range(1, last_module):
			self.network[i].forward(self.network[i-1]._output)
		return self.network[last_module - 1]._output

	def show_updates(self, cpt_epoch):
		print("epoch :", cpt_epoch, end="")
		train_acc = self.score(self._X, self._y, type="accuracy")
		print(", train_acc :", '{:<08g}'.format(train_acc), end="")
		train_loss = self.network[len(self.network) - 1]._output.mean()
		if train_loss < 0:
			print(", train_loss :", '{:<2e}'.format(train_loss), end="")
		else:
			print(", train_loss :", " "+('{:<2e}'.format(train_loss)), end="")
		if self._valid: 
			valid_acc = self.score(self._valid_x, self._valid_y, type="accuracy")
			print(", valid_acc :", '{:<08g}'.format(valid_acc), end="")
			valid_loss = self.network[len(self.network) - 1]._output.mean()
			if valid_loss < 0: 
				print(", valid_loss :", '{:<2e}'.format(valid_loss), end="")
			else:
				print(", valid_loss :", " "+('{:<2e}'.format(valid_loss)), end="")
		print("")

	def update_stats(self):
		self.loss_values.append(self.network[len(self.network) - 1]._output.mean())
		self.acc_values.append(self.score(self._X, self._y, type="accuracy"))

	def score(self, X, y, type="accuracy"):
		if type == "accuracy":
			return np.where(y == self.predict(X).argmax(axis=1), 1, 0).mean()

	def stats(self):
		lv = np.array(self.loss_values)
		plt.plot(lv)
		plt.show()
		la = np.array(self.acc_values)
		plt.plot(la)
		plt.show()

	def summary(self):
		print("=========================================================================")
		print("==> Network :")
		for i, m in enumerate(self.network):
			type_ = str(type(m))
			element = type_.split(".")[-1][:-2]
			if "layers" in type_:
				print("--> Layer", int(i/2), ":", element, "with parameters_shape =", m._parameters.shape, end="")
			elif "activation_functions" in type_:
				print(" and activation :", element)
			elif "loss_functions" in type_:
				print("* Loss :", element)
		print("* Optimizer :", self._optimizer_name)
		print("* Total number of parameters :", self.count_parameters())
		print("=========================================================================")

	def count_parameters(self):
		res = 0
		for m in self.network:
			if "layers" in str(type(m)):
				res += m._parameters.size + m._bias.size
		return res

	def zero_grad(self):
		for m in self.network:
			if "layers" in str(type(m)):
				m.zero_grad()

	def forward(self, X, y):
		last_module = len(self.network) - 1
		## First forward pass on data <self._X>
		self.network[0].forward(X)
		## Forward on all the next layers
		for i in range(1, last_module):
			self.network[i].forward(self.network[i-1]._output)
		## Forward on the loss module which is the last module of the network
		self.network[last_module].forward(y, self.network[last_module - 1]._output)
		## Return the output of the last layer, before the loss module
		return self.network[last_module - 1]._output

	def update_parameters(self, learning_rate):
		for i in range(len(self.network) - 1):
			self.network[i].update_parameters(learning_rate)

	def backward(self):
		loss_function = self.network[len(self.network) - 1]
		## Backward pass on the loss function
		loss_function.backward()
		## Backward pass on every previous layers
		for i in range(len(self.network) - 1, 0, -1):
			self.network[i-1].backward_update_gradient(self.network[i]._new_delta)
			self.network[i-1].backward_delta()