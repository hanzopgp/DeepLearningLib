from Core import *


class MomentumStochasticGradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate, gamma):
		super().__init__()
		self._net = net
		self._net.network.append(loss_function)
		self._learning_rate = learning_rate
		self._gamma = gamma

	def step(self, X, y, n_epochs, verbose):
		n = X.shape[0]
		for cpt_epoch in range(n_epochs):
			self._net.zero_grad()
			for _ in range(n):
				idx = np.random.choice(n)
				x_element, y_element = X[idx].reshape(1, -1), y[idx].reshape(1, -1)
				self._net.forward(x_element, y_element)
				self._net.backward()
				self._net.update_parameters(self._learning_rate, True, self._gamma)
			self._net.update_stats()
			if verbose == True: 
				self._net.show_updates(cpt_epoch=cpt_epoch)
