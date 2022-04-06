from Core import *


class Linear(Module):
	def zero_grad(self):
		pass
	
	def update_parameters(self, learning_rate):
		pass

	def forward(self, input):
		self._input = input
		self._output = self._input
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		self._delta = delta

	def backward_delta(self):
		gradient = np.ones_like(self._input)
		self._new_delta = gradient * self._delta
