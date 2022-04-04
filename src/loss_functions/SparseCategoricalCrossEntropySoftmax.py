from Core import *
from utils.utils import one_hot
from global_imports import *


class SparseCategoricalCrossEntropySoftmax(Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y, yhat.shape[1])
		self._yhat = yhat
		assert(self._y.shape == self._yhat.shape)
		self._yhat = np.where(self._yhat < global_variables.MIN_THRESHOLD, global_variables.MIN_THRESHOLD, self._yhat)
		self._yhat = np.where(self._yhat > global_variables.MAX_THRESHOLD, global_variables.MAX_THRESHOLD, self._yhat)
		sum_ = np.sum(self._y * self._yhat, axis=1)
		log_ = np.log(np.sum(np.exp(self._yhat), axis=1))
		self._output = log_ - sum_

	def backward(self):
		exp_ = np.exp(self._yhat)
		self._new_delta = exp_ / np.sum(exp_, axis=1).reshape((-1, 1)) - self._y