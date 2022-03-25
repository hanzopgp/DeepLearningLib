from Core import *

class Tanh(Module):
	def forward(self, input):
		self._input = input
		self._output = np.tanh(self._input)
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		self._delta = delta
		res = 1 - np.tanh(self._input)**2
		self._gradient = res * self._delta

	def backward_delta(self):
		self._new_delta = self._input * self._delta
