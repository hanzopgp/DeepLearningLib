from sklearn.ensemble import RandomTreesEmbedding
from Core import *
from global_imports import *
from global_variables import *

# Source: https://deepnotes.io/dropout


class Dropout(Module):
	def __init__(self, rate=0.2):
		super().__init__()
		self._rate = rate

	def zero_grad(self):
		pass

	def update_parameters(self, learning_rate):
		pass

	def forward(self, input):
		self._input = input
		## The mask is just a bernouilli scaled by 1/rate
		self._mask = np.random.binomial(1, self._rate, size=self._input.shape) / self._rate
		## To randomly shut down the inputs we can multipy elementwise the input with the mask
		self._output = self._input * self._mask

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		assert(de	lta.shape[0] == self._input.shape[0])
		self._delta = delta
		
	def backward_delta(self):
		## Here we only backpropagate the delta for inputs which weren't shut down during forward pass
		## So we can just multiply elementwise the delta with our mask again
		self._new_delta = self._delta * self._mask