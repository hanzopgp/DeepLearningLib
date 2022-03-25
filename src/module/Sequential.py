from tkinter import Y
from Core import *

from activation_functions.ReLU import ReLU
from activation_functions.Sigmoid import Sigmoid
from activation_functions.Softmax import Softmax
from activation_functions.Tanh import Tanh

class Sequential(Module):
	def __init__(self, input_shape):
		self.network = []
		self._input_shape = input_shape

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

	def compile(self, loss_function, learning_rate):
		self._learning_rate = learning_rate
		self.network.append(loss_function)

	def fit(self, X, y, n_epochs):
		self._y = y
		for _ in range(n_epochs):
			self.forward(X)
			self.backward_update_gradient()
			# self.backward_delta()
			# self.update_parameters()

	def predict(self, input):
		return self.forward(input)

	def summary(self):
		print("=========================================================================")
		for i, m in enumerate(self.network):
			type_ = str(type(m))
			element = type_.split(".")[-1][:-2]
			if "layers" in type_:
				print("Layer", int(i/2), ":", element, "with parameters_shape =", m._parameters.shape, end="")
			elif "activation_functions" in type_:
				print(", Activation :", element)
			elif "loss_functions" in type_:
				print("Loss :", element)
		print("=========================================================================")

	def forward(self, X):
		last_module = len(self.network) - 1
		## First forward pass on data <X>
		self.network[0].forward(X)
		## Forward on all the next layers
		for i in range(1, last_module):
			self.network[i].forward(self.network[i-1]._output)
		## Forward on the loss module which is the last module of the network
		self.network[last_module].forward(self._y, self.network[last_module - 1]._output)
		## Return the output of the last layer, before the loss module
		return self.network[last_module - 1]._output

	def update_parameters(self):
		for module in self.network:
			module.update_parameters(self._learning_rate)

	def backward_update_gradient(self):
		loss_function = self.network[len(self.network) - 1]
		## Backward pass on the loss function
		loss_function.backward(self._y, loss_function._output)
		## Backward pass on every previous layers
		for i in range(len(self.network) - 1, 0, -1):
			self.network[i-1].backward_update_gradient(self.network[i]._delta)

	def backward_delta(self, grad_input, delta):
		last_index = len(self.network) - 1
		self.network[last_index].backward_delta(grad_input, delta)
		for i in range(last_index, 0, -1):
			self.network[i-1].backward_delta(self.network[i]._grad_input, self.network[i]._delta)