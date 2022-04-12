import numpy as np
import nndyi.core


LINEAR_INIT_ARGS = (None, 'xavier')
PARAMS_INIT_FACTOR = 1e-3
OVERFLOW_THRESHOLD = 1e6


class Linear(nndyi.core.Module):
	"""A layer consist of multiple linear neurons, each with its own parameters (weight and bias)"""

	def __init__(self, size_in:int, size_out:int, init='xavier', regularization=1e-9):
		assert size_in > 0
		assert size_out > 0
		assert init in LINEAR_INIT_ARGS
		
		if init is None:
			self._W = np.random.rand(size_in, size_out) * PARAMS_INIT_FACTOR
		else:	# xavier initialization
			self._W = (np.random.rand(size_in, size_out)*2 - 1) / np.sqrt(size_out)
		self._b = np.zeros(size_out)
		self._lambda = regularization
		self.zero_grad()

	def forward(self, data):
		self._input = data
		self._output = data @ self._W + self._b
		# Clip output to avoid overflowing in activation layer
		self._output = np.where(self._output > OVERFLOW_THRESHOLD, OVERFLOW_THRESHOLD, self._output)
		self._output = np.where(self._output < -OVERFLOW_THRESHOLD, -OVERFLOW_THRESHOLD, self._output)
	
	def backward(self):
		self._new_delta = self._delta @ self._W.T

	def zero_grad(self):
		self._grad_W = np.zeros_like(self._W)
		self._grad_b = np.zeros_like(self._b)

	def backward_update_gradient(self, delta):
		self._delta = delta
		self._grad_W += self._input.T @ delta
		self._grad_b += np.sum(delta, axis=0)


class Dropout(nndyi.core.Module):
	"""Drop-out layer with a fixed drop-out rate"""

	def __init__(self, rate=0.2):
		self._rate = rate

	def forward(self, data):
		self._input = data
		# Bernouilli mask scaled by 1/rate
		self._mask = np.random.binomial(1, self._rate, size=self._input.shape) / self._rate
		# To randomly disable the inputs we can multipy elementwise the input with the mask
		self._output = self._input * self._mask

	def backward_update_gradient(self, delta):
		self._delta = delta

	def backward(self):
		# Here we only back propagate the delta for inputs which weren't disabled during forward pass,
		# so we can just multiply elementwise the delta with our mask again
		self._new_delta = self._delta * self._mask
