from Core import *

class MSE(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return (y - yhat) ** 2

	def backward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return 2 * (y - yhat)
