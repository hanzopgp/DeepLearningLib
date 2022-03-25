from Core import *


class MinibatchGradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate, n_batch):
		super().__init__()
		self.net = net
		self.net.network.append(loss_function)
		self.learning_rate = learning_rate
		self.n_batch = n_batch

	def step(self, X, y):
		self.net.zero_grad()
		minibatches_x, minibatches_y = np.array_split(X, self.n_batch), np.array_split(y, self.n_batch)
		for minibatch_x, minibatch_xy in np.random.choice(zip(minibatches_x, minibatches_y)):
			self.net.forward(minibatch_x, minibatch_xy)
			self.net.backward()
			self.net.update_parameters(self.learning_rate)