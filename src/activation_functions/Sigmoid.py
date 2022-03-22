from Core import *

class Sigmoid(Module):
	def forward(self, input):
		return 1/(1 + np.exp(-input))

	def update_parameters(self, learning_rate=1e-3):
		self._parameters -= learning_rate*self._gradient

	def backward_update_gradient(self, input, delta):
		exp_ = np.exp(-input)
		self._gradient += (exp_ / (1 + exp_)**2) + delta
		return self._gradient

	def backward_delta(self, input, delta):
		raise NotImplementedError()
