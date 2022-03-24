from Core import *

class Sigmoid(Module):
	def forward(self, input):
		self._input = input
		return 1/(1 + np.exp(-self._input))

	def backward_update_gradient(self, input, delta):
		exp_ = np.exp(-input)
		self._gradient += (exp_ / (1 + exp_)**2) * delta
		return self._gradient

	def backward_delta(self, input, delta):
		return input * delta
