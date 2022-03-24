from Core import *

class Sequential(Module):
	def __init__(self):
		self.network = []

	def add(self, module):
		self.network.append(module)

	def forward(self, input):
		for module in self.network:
			input = module.forward(input)
		return input

	def backward_update_gradient(self, input, delta):
		self.network[0].backward_update_gradient(input, delta)
		for i in range(1, len(self.network)):
			self.network[i].backward_update_gradient(self.network[i-1]._input, self.network[i-1]last_module._delta)

	def backward_delta(self, input, delta):
		for module in self.network:
			input, delta = module.backward_delta(input, delta)
		return input