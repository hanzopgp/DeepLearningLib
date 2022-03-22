from Core import *

class ReLU(Module):
	def forward(self, input):
		return np.maximum(0, input)

	def update_parameters(self, learning_rate=1e-3):
		self._parameters -= learning_rate*self._gradient

	def backward_update_gradient(self, input, delta):
		self._gradient += np.where(input > 0, 1, 0) @ delta
		return self._gradient

	def backward_delta(self, input, delta):
		return input @ delta
