from Core import *

class ReLU(Activation):
	def forward(self, input):
		return np.maximum(0, input)

	def backward(self, y, yhat):
		pass
