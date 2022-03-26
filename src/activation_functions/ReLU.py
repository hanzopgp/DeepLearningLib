from Core import *

class ReLU(Module):
	def zero_grad(self):
		pass
	
	def update_parameters(self, learning_rate, momentum, gamma):
		pass

	def forward(self, input):
		self._input = input
		self._output = np.maximum(0, self._input)
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		self._delta = delta

	def backward_delta(self):
		gradient = np.where(self._input > 0, 1, 0) * self._delta
		self._new_delta = gradient * self._delta
