from Core import *

class Sigmoid(Activation):
	def forward(self, input):
		return 1/(1 + np.exp(-input))

	def backward(self, y, yhat):
		pass
