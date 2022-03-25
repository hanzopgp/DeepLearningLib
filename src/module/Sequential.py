from Core import *

from activation_functions.ReLU import ReLU
from activation_functions.Sigmoid import Sigmoid
from activation_functions.Softmax import Softmax
from activation_functions.Tanh import Tanh

from loss_functions.MSE import MSE
from loss_functions.BinaryCrossEntropy import BinaryCrossEntropy
from loss_functions.CrossEntropy import CrossEntropy
from loss_functions.SparseBinaryCrossEntropy import SparseBinaryCrossEntropy
from loss_functions.SparseCrossEntropy import SparseCrossEntropy

from matplotlib import pyplot as plt


class Sequential(Module):
	def __init__(self):
		self.network = []
		self.loss_values = []
		self.acc_values = []

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

	def compile(self, loss, learning_rate):
		self._learning_rate = learning_rate
		if loss == "binary_crossentropy":
			self.network.append(BinaryCrossEntropy())
		elif loss == "crossentropy":
			self.network.append(CrossEntropy())
		elif loss == "mse":
			self.network.append(MSE())
		elif loss == "sparse_binary_crossentropy":
			self.network.append(SparseBinaryCrossEntropy())
		elif loss == "sparse_crossentropy":
			self.network.append(SparseCrossEntropy())

	def fit(self, X, y, n_epochs):
		self._y = y
		for _ in range(n_epochs):
			self.forward(X)
			self.backward()
			self.update_parameters()

	def predict(self, input):
		return self.forward(input)

	def score(self, X, y, type="accuracy"):
		if type == "accuracy":
			print("--> Accuracy:", np.where(y == self.predict(X).argmax(axis=0), 1, 0).mean())

	def stats(self):
		lv = np.array(self.loss_values)
		plt.plot(lv)
		plt.show()
		la = np.array(self.acc_values)
		plt.plot(la)
		plt.show()

	def summary(self):
		print("=========================================================================")
		for i, m in enumerate(self.network):
			type_ = str(type(m))
			element = type_.split(".")[-1][:-2]
			if "layers" in type_:
				print("Layer", int(i/2), ":", element, "with parameters_shape =", m._parameters.shape, end="")
			elif "activation_functions" in type_:
				print(" and activation :", element)
			elif "loss_functions" in type_:
				print("Loss :", element)
		print("Total number of parameters :", self.count_parameters())
		print("=========================================================================")

	def count_parameters(self):
		res = 0
		for m in self.network:
			if "layers" in str(type(m)):
				res += m._parameters.size
		return res

	def forward(self, X):
		last_module = len(self.network) - 1
		## First forward pass on data <self._X>
		self.network[0].forward(X)
		## Forward on all the next layers
		for i in range(1, last_module):
			self.network[i].forward(self.network[i-1]._output)
		## Forward on the loss module which is the last module of the network
		self.network[last_module].forward(self._y, self.network[last_module - 1]._output)
		self.loss_values.append(self.network[last_module]._output.mean())
		predictions = self.network[last_module - 1]._output.argmax(axis=1)
		self.acc_values.append(np.where(predictions == self._y, 1, 0).mean())
		## Return the output of the last layer, before the loss module
		return self.network[last_module - 1]._output

	def update_parameters(self):
		for i in range(len(self.network) - 1):
			self.network[i].update_parameters(self._learning_rate)

	def backward(self):
		loss_function = self.network[len(self.network) - 1]
		## Backward pass on the loss function
		loss_function.backward()
		## Backward pass on every previous layers
		for i in range(len(self.network) - 1, 0, -1):
			self.network[i-1].backward_update_gradient(self.network[i]._new_delta)
			self.network[i-1].backward_delta()

	# def backward_delta(self, grad_input, delta):
	# 	last_index = len(self.network) - 1
	# 	## Backward delta on the loss function
	# 	self.network[last_index].backward_delta(grad_input, delta)
	# 	## Backward delta on every previous layers but the first one
	# 	for i in range(last_index, 1, -1):
	# 		self.network[i-1].backward_delta(self.network[i]._grad_input, self.network[i].new__delta)