from Core import *

class Softmax(Activation):
	def forward(self, input):
		return np.exp(input) / np.sum(np.exp(input))

	def backward(self, y, yhat):
		pass
