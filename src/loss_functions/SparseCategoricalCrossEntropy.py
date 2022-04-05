from Core import *
from utils.utils import one_hot

# Source: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy


class SparseCategoricalCrossEntropy(Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y, yhat.shape[1])
		self._yhat = yhat
		# assert(self._y.shape == self._yhat.shape)
		self._output = 1 - np.sum(self._yhat * self._y, axis=1)
		# eps = 1e-100
		# self._output = -self._y * np.log(self._yhat+eps)

	def backward(self):
		self._new_delta = self._yhat - self._y
		# eps = 1e-100
		# self._new_delta = -self._y/(self._yhat+eps)