from Core import *
import random

class StochasticGradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate):
		super().__init__()
		self.net = net
		self.net.network.append(loss_function)
		self.learning_rate = learning_rate

	def step(self, X, y):
		n = X.shape[0]
		self.net.zero_grad()
		for _ in range(n):
			idx = np.random.choice(n)
			x_element, y_element = X[idx].reshape(1, -1), y[idx].reshape(1, -1)
			self.net.forward(x_element, y_element)
			self.net.backward()
			self.net.update_parameters(self.learning_rate)
