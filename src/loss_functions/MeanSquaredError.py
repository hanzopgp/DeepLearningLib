from Core import *
from utils.utils import one_hot


class MeanSquaredError(Loss):
	def forward(self, y, yhat):
		self._y = y
		self._yhat = yhat
		assert(self._y.shape == yhat.shape)
		self._output = np.sum((self._y - self._yhat) ** 2, axis=1)

	def backward(self):
		self._new_delta = 2 * (self._y - self._yhat)
