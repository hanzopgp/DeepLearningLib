from Core import *

# Source: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html


class Linear(Module):
	def __init__(self, size_in, size_out, init_type="xavier"):
		super().__init__()
		norm = np.sqrt(size_out)
		if init_type == "xavier":
			self._parameters = (np.random.rand(size_in, size_out) * 2 - 1) / norm
		else:
			self._parameters = np.random.rand(size_in, size_out) * 1e-3
		self._bias = np.zeros(size_out)
		self.zero_grad()

	def zero_grad(self):
		self._gradient = np.zeros_like(self._parameters)
		self._gradient_bias = np.zeros_like(self._bias)

	def update_parameters(self, learning_rate):
		self._parameters -= learning_rate * self._gradient
		self._bias -= learning_rate * self._gradient_bias

	def forward(self, input):
		assert(input.shape[1] == self._parameters.shape[0])
		self._input = input
		self._output = self._input @ self._parameters + self._bias

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		assert(delta.shape[0] == self._input.shape[0])
		self._delta = delta
		self._gradient += self._input.T @ self._delta 
		## THERE IS A PROBLEM WHEN USING GRADIENT DESCENT, THERE SHOULD BE A MEAN() OR SUM() SOMEHWERE
		## BECAUSE GRADIENT IS WAY TOO HIGH WHEN ON WHOLE BATCH
		# print("INPUT",self._input)
		# print("DELTA",self._delta)
		self._gradient_bias += self._delta.sum(axis=0)
		assert(self._gradient.shape == self._parameters.shape)
		assert(self._gradient_bias.shape == self._bias.shape)
		
	def backward_delta(self):
		self._new_delta = self._delta @ self._parameters.T	