from Core import *

# Source: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

class BinaryCrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return -y*np.log(yhat) + (1-y)*np.log(1-yhat)

	def backward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return (yhat - y)/ yhat / (1-yhat)