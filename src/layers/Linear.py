from Core import *

class Linear(Module):
	def __init__(self, size_in, size_out):
		super().__init__()
		self._parameters = np.random.rand(size_in, size_out)
		self._bias = np.random.rand(size_out)
		self.zero_grad()

	def zero_grad(self):
		self._gradient = np.zeros_like(self._parameters)

	def forward(self, input):
		self._output = input @ self._parameters + self._bias
		return self._output

	def backward_update_gradient(self, input, delta):
		self._delta = delta
		self._gradient += input.T * delta 
		return self._gradient
		
	def backward_delta(self, input, delta):
		return input @ self._parameters * delta
