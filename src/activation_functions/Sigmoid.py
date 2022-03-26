from Core import *

class Sigmoid(Module):
	def zero_grad(self):
		pass
	
	def update_parameters(self, learning_rate, momentum, gamma):
		pass

	def forward(self, input):
		self._input = input
		self._output = 1/(1 + np.exp(-self._input))
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		self._delta = delta

	def backward_delta(self):
		sigmoid = 1/(1 + np.exp(-self._input))
		gradient = sigmoid * (1 - sigmoid)
		self._new_delta = gradient * self._delta
