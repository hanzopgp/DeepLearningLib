import numpy as np
from collections import namedtuple
import core
import layer


EarlyStopping = namedtuple("EarlyStopping", ('metric', 'min_delta', 'patience'))


class GradientDescent(core.Optimizer):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		super()._make_es_handler(early_stopping)
		for cpt_epoch in range(n_epochs):
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
		if isinstance(l, layer.Linear):
			l._W -= (self._lr * l._grad_W) - (l._lambda * l._W)
			l._b -= (self._lr * l._grad_b) - (l._lambda * l._b)
			l.zero_grad()


class StochasticGradientDescent(core.Optimizer):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		super()._make_es_handler(early_stopping)


class MinibatchGradientDescent(core.Optimizer):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		super()._make_es_handler(early_stopping)


class Adam(core.Optimizer):
	def step(self, X, y, n_epochs, verbose, early_stopping):
		super()._make_es_handler(early_stopping)

