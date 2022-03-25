from Core import *

class Tanh(Module):
	def zero_grad(self):
		pass

	def update_parameters(self, learning_rate):
		pass

	def forward(self, input):
		self._input = input
		self._output = np.tanh(self._input)
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		print(delta.shape)
		print(self._output.shape)
		assert(delta.shape == self._output.shape)
		self._delta = delta

	def backward_delta(self):
		gradient = 1 - np.tanh(self._input)**2
		self._new_delta = gradient * self._delta
