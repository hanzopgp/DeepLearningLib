from Core import *
from global_imports import *
from global_variables import *


# Source: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right


class BinaryCrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		self._yhat = np.where(self._yhat < global_variables.MIN_THRESHOLD, global_variables.MIN_THRESHOLD, self._yhat)
		self._yhat = np.where(self._yhat > global_variables.MAX_THRESHOLD, global_variables.MAX_THRESHOLD, self._yhat)
		self._output = -self._y*np.log(self._yhat+DIVIDE_BY_ZERO_EPS) + (1-self._y)*np.log(1-self._yhat+DIVIDE_BY_ZERO_EPS)

	def backward(self):
		self._new_delta = ((1 - self._y) / (1 - self._yhat+DIVIDE_BY_ZERO_EPS)) - (self._y / self._yhat+DIVIDE_BY_ZERO_EPS)
		# self._new_delta = -(self._y / self._yhat) + (-1 + self._y) / (1 - self._yhat)
		# self._new_delta = ((self._yhat - self._y) / self._yhat) / (1-self._yhat+eps)