from Core import *

class StochasticGradientDescent(Optimizer):
	def __init__(self, net, loss, learning_rate):
		super().__init__()
		self.net = net
		self.loss = loss
		self.learning_rate = learning_rate

	def step(self, batch_x, batch_y):
		# self.net.forward(batch_x)
		# self.net.backward()
		# self.net.update_parameters()
		# compute network output thanks to batch_x
		# compute loss thanks to batch_y and previous computation
		# execute backward pass
		# update parameters
		pass
