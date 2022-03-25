from Core import *

class MSE(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		self._output = (self._y - self._yhat) ** 2

	def backward(self):
		assert(self._y.shape == self._yhat.shape)
		self._new_delta = 2 * (self._y - self._yhat)
