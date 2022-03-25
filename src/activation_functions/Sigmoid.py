from Core import *

class Sigmoid(Module):
	def forward(self, input):
		self._input = input
		self._output = 1/(1 + np.exp(-self._input))
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		exp_ = np.exp(-self._input)
		self._gradient += (exp_ / (1 + exp_)**2) * delta

	def backward_delta(self):
		self._new_delta = self._input * self._delta
