from Core import *
from utils.utils import *

# Source: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

class SparseBinaryCrossEntropy(Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y)
		assert(self._y.shape == yhat.shape)
		self._output = -self._y*np.log(yhat) + (1-self._y)*np.log(1-yhat)

	def backward(self, y, yhat):
		assert(self._y.shape == yhat.shape)
		self._delta = (yhat - self._y) / yhat / (1-yhat)