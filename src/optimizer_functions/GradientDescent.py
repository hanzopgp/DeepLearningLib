from Core import *
from global_imports import *


class GradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate):
		super().__init__()
		self._net = net
		self._net.network.append(loss_function)
		self._learning_rate = learning_rate

	def step(self, X, y, n_epochs, verbose):
		for cpt_epoch in range(n_epochs):
			self._net.zero_grad()
			self._net.forward(X, y)
			self._net.backward()
			self._net.update_parameters(self._learning_rate)
			self._net.update_stats()
			if verbose == True: 
				self._net.show_updates(cpt_epoch=cpt_epoch)