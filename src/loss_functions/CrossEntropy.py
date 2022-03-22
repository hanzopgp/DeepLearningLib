from Core import *

# Source: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy

class CrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return -y * np.log(yhat)

	def backward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return -y/yhat