from Core import *

# Source: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html


class Linear(Module):
	def __init__(self, size_in, size_out):
		super().__init__()
		self._parameters = np.random.rand(size_in, size_out)
		self._bias = np.random.rand(size_out)
		self.zero_grad()

	def zero_grad(self):
		self._gradient = np.zeros_like(self._parameters)

	def forward(self, input):
		print(input.shape)
		print(self._parameters.shape)
		assert(input.shape[1] == self._parameters.shape[0])
		self._input = input
		self._output = self._input @ self._parameters + self._bias

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._gradient.shape)
		assert(delta.shape == self._output.shape)
		self._delta = delta
		self._gradient += self._input * self._delta 
		assert(self._gradient.shape == self._parameters.shape)
		
	def backward_delta(self):
		self._new_delta = self._parameters * self._delta
