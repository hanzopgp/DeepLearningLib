from Core import *

class Tanh(Activation):
	def forward(self, input):
		return np.tanh(input)

	def backward(self, y, yhat):
		pass
