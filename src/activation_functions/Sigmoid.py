from Core import *

class Sigmoid(Module):
	def forward(self, input):
		self._input = input
		self._output = 1/(1 + np.exp(-self._input))

	def backward_update_gradient(self, input, delta):
		exp_ = np.exp(-input)
		self._gradient += (exp_ / (1 + exp_)**2) * delta

	def backward_delta(self, input, delta):
		self._delta = input * delta
