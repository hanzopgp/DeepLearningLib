from Core import *
from global_imports import *
from global_variables import *

# Source: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html


class Linear(Module):
	def __init__(self, size_in, size_out, init_type="xavier", regularization_lambda=1e-9):
		super().__init__()
		if init_type == "xavier":
			self._parameters = (np.random.rand(size_in, size_out) * 2 - 1) / np.sqrt(size_out)
		else:
			self._parameters = np.random.rand(size_in, size_out) * global_variables.PARAMETERS_INIT_FACTOR
		self._regularization_lambda = regularization_lambda
		self._bias = np.zeros(size_out)
		self.zero_grad()

	def zero_grad(self):
		self._gradient = np.zeros_like(self._parameters)
		self._gradient_bias = np.zeros_like(self._bias)

	def update_parameters(self, learning_rate):
		self._parameters -= (learning_rate * self._gradient) - (self._regularization_lambda * self._parameters)
		self._bias -= (learning_rate * self._gradient_bias) - (self._regularization_lambda * self._bias)

	def forward(self, input):
		assert(input.shape[1] == self._parameters.shape[0])
		self._input = input
		self._output = self._input @ self._parameters + self._bias
		## Avoiding overflow in activation functions
		# self._output = np.where(self._output > AVOID_OVERFLOW_VALUE, AVOID_OVERFLOW_VALUE, self._output)
		# self._output = np.where(self._output < -AVOID_OVERFLOW_VALUE, -AVOID_OVERFLOW_VALUE, self._output)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		assert(delta.shape[0] == self._input.shape[0])
		self._delta = delta
		self._gradient += self._input.T @ self._delta 
		self._gradient_bias += self._delta.sum(axis=0)
		assert(self._gradient.shape == self._parameters.shape)
		assert(self._gradient_bias.shape == self._bias.shape)
		
	def backward_delta(self):
		self._new_delta = self._delta @ self._parameters.T	



## Tried to implement numba but it doesn't work with class atm
# 	def forward(self, input):
# 		assert(input.shape[1] == self._parameters.shape[0])
# 		self._input = input
# 		self._output = _forward(self._input, self._parameters, self._bias)

# 	def backward_update_gradient(self, delta):
# 		assert(delta.shape == self._output.shape)
# 		assert(delta.shape[0] == self._input.shape[0])
# 		self._delta = delta
# 		a, b = _backward_update_gradient(self._delta, self._input)
# 		self._gradient += a
# 		self._gradient_bias += b
# 		assert(self._gradient.shape == self._parameters.shape)
# 		assert(self._gradient_bias.shape == self._bias.shape)
		
# 	def backward_delta(self):
# 		self._new_delta = backward_delta(self._delta, self._parameters)


# @jit(nopython=True, parallel=True, fastmath=True)
# def _forward(input, parameters, bias):
# 		return input @ parameters + bias

# @jit(nopython=True, parallel=True, fastmath=True)
# def _backward_update_gradient(delta, input):
# 		return input.T @ delta, delta.sum(axis=0)

# @jit(nopython=True, parallel=True, fastmath=True)
# def backward_delta(delta, parameters):
# 		return delta @ parameters.T