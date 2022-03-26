from Core import *


class RootMeanSquaredError(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		self._output = np.sum(np.sqrt((self._y - self._yhat) ** 2), axis=1)

	def backward(self):
		mse = (self._y - self._yhat) ** 2
		d_mse = 2 * (self._y - self._yhat)
		self._new_delta = 1/(2*mse) * (d_mse)
