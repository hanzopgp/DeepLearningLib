from Core import *

class Tanh(Module):
	def forward(self, input):
		return np.tanh(input)

	def update_parameters(self, learning_rate=1e-3):
		self._parameters -= learning_rate*self._gradient

	def backward_update_gradient(self, input, delta):
		self._gradient = 1 - np.tanh(input)**2 + delta
		return self._gradient

	def backward_delta(self, input, delta):
		raise NotImplementedError()
