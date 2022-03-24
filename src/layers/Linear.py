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
		# print(input.shape)
		# print(self._parameters.shape)
		assert(input.shape[1] == self._parameters.shape[0])
		assert(self._parameters.shape[1] == self._bias.shape[0])
		self._input = input
		self._output = self._input @ self._parameters + self._bias

	def backward_update_gradient(self, grad_input, delta):
		self._grad_input = grad_input
		self._gradient += self._grad_input.T * self._delta 
		
	def backward_delta(self, input, delta):
		self._delta = self._grad_input @ self._parameters * self._delta
