from Core import *

# Source: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy


class CategoricalCrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		self._output = 1 - np.sum(self._yhat * self._y, axis=1)
		# eps = 1e-100
		# self._output = -self._y * np.log(self._yhat+eps)

	def backward(self):
		self._new_delta = self._yhat - self._y
		# eps = 1e-100
		# self._new_delta = -self._y / (self._yhat+eps)