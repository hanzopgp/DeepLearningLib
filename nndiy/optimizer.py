import numpy as np
import nndiy.core
import nndiy.layer
import nndiy.convolution


class GradientDescent(nndiy.core.Optimizer):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		for cpt_epoch in range(1, n_epochs + 1):
			self._seq._forward(X, y)
			self._seq._backward()
			for l in self._seq._net:
				self._update_layer_params(l)
			stats = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, *stats)
			
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

			if self._es.update(cpt_epoch, stats):
				print("--> Early stopping triggered and best model returned from epoch number", self._es.best_cpt_epoch)
				break

	def _update_layer_params(self, l):
		if isinstance(l, nndiy.layer.Linear) or isinstance(l, nndiy.convolution.Convo1D):
			l._W -= (self._lr * l._grad_W) - (l._lambda * l._W)
			l._b -= (self._lr * l._grad_b) - (l._lambda * l._b)
			l.zero_grad()


class StochasticGradientDescent(GradientDescent):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		n = X.shape[0]
		idx_order = np.arange(n)
		for cpt_epoch in range(1, n_epochs + 1):
			np.random.shuffle(idx_order)
			for i in idx_order:
				x_elem, y_elem = X[i].reshape(1, -1), y[i].reshape(1, -1)
				self._seq._forward(x_elem, y_elem)
				self._seq._backward()
				for l in self._seq._net:
					self._update_layer_params(l)
			stats = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, *stats)
			
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

			if self._es.update(cpt_epoch, stats):
				print("--> Early stopping triggered and best model returned from epoch number", self._es.best_cpt_epoch)
				break


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
			stats = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, *stats)
			
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

			if self._es.update(cpt_epoch, stats):
				print("--> Early stopping triggered and best model returned from epoch number", self._es.best_cpt_epoch)
				break

	def _make_batches(self, X:np.ndarray, y:np.ndarray):
		batch_sz = X.shape[0] // self._n_batch
		batch_X = []
		batch_y = []
		for i in range(self._n_batch):
			batch_X.append(X[i*batch_sz:(i+1)*batch_sz])
			batch_y.append(y[i*batch_sz:(i+1)*batch_sz])
		return np.array(batch_X), np.array(batch_y)


class Adam(nndiy.core.Optimizer):
	def __init__(self, network, learning_rate, decay, n_batch, b1=0.9, b2=0.999, eps=1e-8):
		super().__init__(network, learning_rate, decay, n_batch)
		self._mw = []
		self._vw = []
		self._mb = []
		self._vb = []
		self._b1 = b1
		self._b2 = b2
		self._eps = eps
		self._cpt_layer = 0
		self._init_shapes = [False, False, False]

	def step(self, X, y, n_epochs, verbose, early_stopping):
		self._make_es_handler(early_stopping)
		n = X.shape[0]
		idx_order = np.arange(n)
		for cpt_epoch in range(1, n_epochs + 1):
			np.random.shuffle(idx_order)
			for i in idx_order:
				x_elem, y_elem = X[i].reshape(1, -1), y[i].reshape(1, -1)
				self._seq._forward(x_elem, y_elem)
				self._seq._backward()
				for l in self._seq._net:
					self._update_layer_params(l, cpt_epoch)
				# Reset layer index
				self._cpt_layer = 0
			stats = self._seq._update_stats()
			if verbose:
				self._show_updates(cpt_epoch, *stats)
			
			# Recalculate learning rate using decay and number of epoch
			self._lr *= (1 / (1 + self._decay * cpt_epoch))

			if self._es.update(cpt_epoch, stats):
				print("--> Early stopping triggered and best model returned from epoch number", self._es.best_cpt_epoch)
				break

	def _update_layer_params(self, l, cpt_epoch):
		if isinstance(l, nndiy.layer.Linear) or isinstance(l, nndiy.convolution.Convo1D):
			# Init if not done
			if not self._init_shapes[self._cpt_layer]:
				self._mw.append(np.zeros_like(l._grad_W))
				self._vw.append(np.zeros_like(l._grad_W))
				self._mb.append(np.zeros_like(l._grad_b))
				self._vb.append(np.zeros_like(l._grad_b))
				self._init_shapes[self._cpt_layer] = True

			# Compute momentums
			self._mw[self._cpt_layer] = self._b1 * self._mw[self._cpt_layer] + (1 - self._b1) * l._grad_W
			self._vw[self._cpt_layer] = self._b2 * self._vw[self._cpt_layer] + (1 - self._b2) * np.power(l._grad_W, 2)
			self._mb[self._cpt_layer] = self._b1 * self._mb[self._cpt_layer] + (1 - self._b1) * l._grad_b
			self._vb[self._cpt_layer] = self._b2 * self._vb[self._cpt_layer] + (1 - self._b2) * np.power(l._grad_b, 2)
			
			# Compute corrections
			mw_hat = self._mw[self._cpt_layer] / (1 - self._b1 ** cpt_epoch)
			vw_hat = self._vw[self._cpt_layer] / (1 - self._b2 ** cpt_epoch)
			mb_hat = self._mb[self._cpt_layer] / (1 - self._b1 ** cpt_epoch)
			vb_hat = self._vb[self._cpt_layer] / (1 - self._b2 ** cpt_epoch)
			
			# Update parameters
			w_update = self._lr * mw_hat / np.where(vw_hat > 0, vw_hat**0.5, self._eps)
			b_update = self._lr * mb_hat / np.where(vb_hat > 0, vb_hat**0.5, self._eps)
			l._W -= w_update
			l._b -= b_update

			# Reset gradients after update
			l.zero_grad()

			# Go to next layer index
			self._cpt_layer += 1


