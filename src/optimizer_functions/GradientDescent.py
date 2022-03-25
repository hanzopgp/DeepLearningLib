from Core import *

class GradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate):
		super().__init__()
		self.net = net
		self.net.network.append(loss_function)
		self.learning_rate = learning_rate

	def step(self, batch_x, batch_y):
		self.net.forward(batch_x, batch_y)
		self.net.backward()
		self.net.update_parameters(self.learning_rate)