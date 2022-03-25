from Core import *

# Source: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

class BinaryCrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		self._y = y
		self._yhat = yhat
		eps = 1e-100
		self._output = -self._y*np.log(self._yhat+eps) + (1-self._y)*np.log(1-self._yhat+eps)

	def backward(self):
		assert(self._y.shape == self._yhat.shape)
		eps = 1e-100
		self._new_delta = ((self._yhat - self._y)/ self._yhat) / (1-self._yhat+eps)