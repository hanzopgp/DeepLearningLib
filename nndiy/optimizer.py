import numpy as np
import nndiy.core
import nndiy.layer


class GradientDescent(nndiy.core.Optimizer):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		for cpt_epoch in range(1, n_epochs + 1):
			self._seq._forward(X, y)
			self._seq._backward()
			for l in self._seq._net:
				self._update_layer_params(l)
			train_loss, train_acc, valid_loss, valid_acc = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, train_loss, train_acc, valid_loss, valid_acc)
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

	def _update_layer_params(self, l):
		if isinstance(l, nndiy.layer.Linear):
			l._W -= (self._lr * l._grad_W) - (l._lambda * l._W)
			l._b -= (self._lr * l._grad_b) - (l._lambda * l._b)
			l.zero_grad()


class StochasticGradientDescent(GradientDescent):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		n = X.shape[0]
		for cpt_epoch in range(1, n_epochs + 1):
			for _ in range(n):
				i = np.random.randint(n)
				x_elem, y_elem = X[i].reshape(1, -1), y[i].reshape(1, -1)
				self._seq._forward(x_elem, y_elem)
				self._seq._backward()
				for l in self._seq._net:
					self._update_layer_params(l)
			train_loss, train_acc, valid_loss, valid_acc = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, train_loss, train_acc, valid_loss, valid_acc)
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))


class MinibatchGradientDescent(GradientDescent):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		batches_X, batches_y = self._make_batches(X, y)
		n = batches_X.shape[0]
		for cpt_epoch in range(1, n_epochs + 1):
			for _ in range(n):
				i = np.random.randint(n)
				batch_X, batch_y = batches_X[i], batches_y[i]
				self._seq._forward(batch_X, batch_y)
				self._seq._backward()
				for l in self._seq._net:
					self._update_layer_params(l)
			train_loss, train_acc, valid_loss, valid_acc = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, train_loss, train_acc, valid_loss, valid_acc)
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

	def _make_batches(self, X:np.ndarray, y:np.ndarray):
		batch_sz = X.shape[0] // self._n_batch
		batch_X = []
		batch_y = []
		for i in range(self._n_batch):
			batch_X.append(X[i*batch_sz:(i+1)*batch_sz])
			batch_y.append(y[i*batch_sz:(i+1)*batch_sz])
		return np.array(batch_X), np.array(batch_y)


class Adam(nndiy.core.Optimizer):
	def __init__(self, network, learning_rate, decay, n_batch, b1=0.9, b2=0.999, alpha=1e-3, eps=1e-8):
		super().__init__(network, learning_rate, decay, n_batch)
		self._mw = 0
		self._vw = 0
		self._mb = 0
		self._vb = 0
		self._b1 = b1
		self._b2 = b2
		self._alpha = alpha
		self._eps = eps

	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		n = X.shape[0]
		for cpt_epoch in range(1, n_epochs + 1):
			for _ in range(n):
				i = np.random.randint(n)
				x_elem, y_elem = X[i].reshape(1, -1), y[i].reshape(1, -1)
				self._seq._forward(x_elem, y_elem)
				self._seq._backward()
				for l in self._seq._net:
					try:
						self._update_layer_params(l, cpt_epoch)
					except:
						continue
			train_loss, train_acc, valid_loss, valid_acc = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, train_loss, train_acc, valid_loss, valid_acc)
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

	def _update_layer_params(self, l:nndiy.layer.Linear, cpt_epoch:int):
		## Compute momentums
		self._mw = self._b1 * self._mw + (1 - self._b1) * l._grad_W
		self._vw = self._b2 * self._vw + (1 - self._b2) * np.power(l._grad_W, 2)
		self._mb = self._b1 * self._mb + (1 - self._b1) * l._grad_b
		self._vb = self._b2 * self._vb + (1 - self._b2) * np.power(l._grad_b, 2)
		## Compute corrections
		mw_hat = self._mw / (1 - self._b1 ** cpt_epoch)
		vw_hat = self._vw / (1 - self._b2 ** cpt_epoch)
		mb_hat = self._mb / (1 - self._b1 ** cpt_epoch)
		vb_hat = self._vb / (1 - self._b2 ** cpt_epoch)
		## Compute update value
		w_update = self._lr * mw_hat / (np.sqrt(np.where(vw_hat > 0, vw_hat, 0)) + self._eps)
		b_update = self._lr * mb_hat / (np.sqrt(np.where(vb_hat > 0, vb_hat, 0)) + self._eps)
		## Update parameters
		l._W -= w_update
		l._b -= b_update
		## Reset gradients
		l.zero_grad()