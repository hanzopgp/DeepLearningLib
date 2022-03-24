from Core import *

class ReLU(Module):
	def forward(self, input):
		self._input = input
		self._output = np.maximum(0, self._input)

	def backward_update_gradient(self, input, delta):
		self._gradient += np.where(input > 0, 1, 0) * delta

	def backward_delta(self, input, delta):
		self._delta = input * delta
