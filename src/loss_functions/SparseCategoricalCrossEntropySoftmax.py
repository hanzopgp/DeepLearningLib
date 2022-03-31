from Core import *
from utils.utils import one_hot


class SparseCategoricalCrossEntropySoftmax(Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y, yhat.shape[1])
		self._yhat = yhat
		assert(self._y.shape == self._yhat.shape)
		eps = 1e-100
		sum_ = np.sum(self._y * self._yhat, axis=1)
		log_ = np.log(np.sum(np.exp(self._yhat), axis=1)+eps)
		self._output = log_ - sum_

	def backward(self):
		exp_ = np.exp(self._yhat)
		self._new_delta = exp_ / np.sum(exp_, axis=1).reshape((-1,1)) - self._y