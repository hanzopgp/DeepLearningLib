from Core import *
from utils.utils import one_hot
from global_imports import *

# Source: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right


class SparseBinaryCrossEntropy(Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y, yhat.shape[1])
		self._yhat = yhat
		assert(self._y.shape == self._yhat.shape)
		self._yhat = np.where(self._yhat < global_variables.MIN_THRESHOLD, global_variables.MIN_THRESHOLD, self._yhat)
		self._yhat = np.where(self._yhat > global_variables.MAX_THRESHOLD, global_variables.MAX_THRESHOLD, self._yhat)
		self._output = - self._y*np.log(self._yhat) + (1-self._y)*np.log(1-self._yhat)

	def backward(self):
		self._new_delta = ((1 - self._y) / (1 - self._yhat)) - (self._y / self._yhat)
		# self._new_delta = ((self._yhat - self._y) / self._yhat) / (1-self._yhat+eps)