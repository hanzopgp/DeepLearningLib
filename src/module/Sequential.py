from Core import *

class Sequential(Module):
	def __init__(self):
		self.network = []

	def add(self, module):
		self.network.append(module)

	def forward(self, input):
		self.network[0].forward(input)
		for i in range(1, len(self.network)):
			self.network[i].forward(self.network[i-1]._input)

	def update_parameters(self, learning_rate=0.001):
		for module in self.network:
			module.update_parameters(learning_rate)

	def backward_update_gradient(self, grad_input, delta):
		last_index = len(self.network) - 1
		self.network[last_index].backward_update_gradient(grad_input, delta)
		for i in range(last_index, 0, -1):
			self.network[i-1].backward_update_gradient(self.network[i]._grad_input, self.network[i]._delta)

	def backward_delta(self, grad_input, delta):
		last_index = len(self.network) - 1
		self.network[last_index].backward_delta(grad_input, delta)
		for i in range(last_index, 0, -1):
			self.network[i-1].backward_delta(self.network[i]._grad_input, self.network[i]._delta)