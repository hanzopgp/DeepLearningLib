from Core import *

class ReLU(Module):
	def forward(self, input):
		self._input = input
		self._output = np.maximum(0, self._input)
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		self._delta = delta
		self._gradient += np.where(self._input > 0, 1, 0) * self._delta

	def backward_delta(self):
		self._new_delta = self._input * self._delta
