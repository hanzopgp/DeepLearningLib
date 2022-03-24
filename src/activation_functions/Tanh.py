from Core import *

class Tanh(Module):
	def forward(self, input):
		self._input = input
		self._output = np.tanh(self._input)

	def backward_update_gradient(self, input, delta):
		res = 1 - np.tanh(input)**2
		self._gradient = res * delta

	def backward_delta(self, input, delta):
		self._delta = input * delta
