from Core import *
from utils.utils import *

# Source: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

class SparseBinaryCrossEntropy(Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y)
		self._yhat = yhat
		assert(self._y.shape == self._yhat.shape)
		eps = 1e-100
		self._output = -self._y*np.log(self._yhat+eps) + (1-self._y)*np.log(1-self._yhat+eps)

	def backward(self):
		assert(self._y.shape == self._yhat.shape)
		self._new_delta = ((self._yhat - self._y) / self._yhat) / (1-self._yhat)