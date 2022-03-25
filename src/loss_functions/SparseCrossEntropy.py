from Core import *
from utils.utils import *

# Source: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy

class SparseCrossEntropy(Loss):
	def forward(self, y, yhat):
		assert(y.shape == yhat.shape)
		yhat = one_hot(yhat)
		return -y * np.log(yhat)

	def backward(self, y, yhat):
		assert(y.shape == yhat.shape)
		return -y/yhat