from Core import *
from global_imports import *


class MinibatchGradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate, n_batch):
		super().__init__()
		self._net = net
		self._net._network.append(loss_function)
		self._learning_rate = learning_rate
		self._n_batch = n_batch

	def step(self, X, y, n_epochs, verbose, early_stopping):
		minibatches_x, minibatches_y = self.build_minibatches(X, y)
		n = minibatches_x.shape[0]
		for cpt_epoch in range(n_epochs):
			for _ in range(n):
			#for _ in tqdm(range(n)):
				idx = np.random.randint(n)
				minibatch_x, minibatch_y = minibatches_x[idx], minibatches_y[idx]
				self._net.forward(minibatch_x, minibatch_y)
				self._net.backward()
				self._net.update_parameters(self._learning_rate)
			self._net.update_stats()
			if verbose == True: 
				self._net.show_updates(cpt_epoch, self._learning_rate)

	def build_minibatches(self, X, y):
		minibatch_size = X.shape[0] // self._n_batch
		minibatch_X = []
		minibatch_y = []
		for i in range(self._n_batch):
			minibatch_X.append(X[i*minibatch_size:(i+1)*minibatch_size])
			minibatch_y.append(y[i*minibatch_size:(i+1)*minibatch_size])
		return np.array(minibatch_X), np.array(minibatch_y)
