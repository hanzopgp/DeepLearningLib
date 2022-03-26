from Core import *



class MeanAbsoluteError(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		self._output = np.abs(self._y - self._yhat)

	def backward(self):
		self._new_delta = np.where(self._yhat > self._y, 1, -1)
