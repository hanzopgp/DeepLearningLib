from Core import *


class MinibatchGradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate, n_batch):
		super().__init__()
		self._net = net
		self._net.network.append(loss_function)
		self._learning_rate = learning_rate
		self._n_batch = n_batch

	def step(self, X, y):
		self._net.zero_grad()
		minibatches_x, minibatches_y = np.array_split(X, self._n_batch), np.array_split(y, self._n_batch)
		for minibatch_x, minibatch_xy in np.random.choice(zip(minibatches_x, minibatches_y)):
			self._net.forward(minibatch_x, minibatch_xy)
			self._net.backward()
			self._net.update_parameters(self._learning_rate)