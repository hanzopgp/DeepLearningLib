from Core import *

# Source: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy

class CrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		eps = 1e-100
		self._output = -self._y * np.log(self._yhat+eps)

	def backward(self):
		assert(self._y.shape == self._yhat.shape)
		eps = 1e-100
		self._new_delta = -self._y / (self._yhat+eps)