from Core import *
from global_imports import *


class MinibatchGradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate, n_batch):
		super().__init__()
		self._net = net
		self._net.network.append(loss_function)
		self._learning_rate = learning_rate
		self._n_batch = n_batch

	def step(self, X, y, n_epochs, verbose):
		minibatches_x, minibatches_y = np.array(np.array_split(X, self._n_batch)), np.array(np.array_split(y, self._n_batch))
		n = minibatches_x.shape[0]
		for cpt_epoch in range(n_epochs):
			self._net.zero_grad()
			for _ in tqdm(range(n)):
				idx = np.random.choice(n)
				minibatch_x, minibatch_y = minibatches_x[idx], minibatches_y[idx]
				self._net.forward(minibatch_x, minibatch_y)
				self._net.backward()
				self._net.update_parameters(self._learning_rate)
			self._net.update_stats()
			if verbose == True: 
				self._net.show_updates(cpt_epoch=cpt_epoch)