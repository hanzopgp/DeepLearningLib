from Core import *

class RootMeanSquaredError(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		self._output = np.sqrt((self._y - self._yhat) ** 2)

	def backward(self):
		assert(self._y.shape == self._yhat.shape)
		mse = (self._y - self._yhat) ** 2
		d_mse = 2 * (self._y - self._yhat)
		self._new_delta = 1/(2*mse) * (d_mse)
