import numpy as np
from collections import namedtuple
from typing import Tuple

ES_METRICS = ('valid_loss', 'train_loss', 'valid_accuracy', 'train_accuracy')


EarlyStopping = namedtuple("EarlyStopping", ('metric', 'min_delta', 'patience'))


class EarlyStoppingHandler():
	def __init__(self, es:EarlyStopping, model):
		assert es.metric in ES_METRICS

		if es.metric == 'valid_loss':
			self.update = self._update_valid_loss
		elif es.metric == 'train_loss':
			self.update = self._update_train_loss
		elif es.metric == 'valid_accuracy':
			self.update = self._update_valid_accu
		else:
			self.update = self._update_train_accu

		self.es = es
		self.best_model = model

		self.best_ep = 0
		self.best_valid_loss = np.inf
		self.best_train_loss = np.inf
		self.best_valid_acc = 0.
		self.best_train_acc = 0.
		self.cpt_patience = 0

	def _update_valid_loss(self, ep:int, stats:Tuple[float,float,float,float]) -> bool:
		valid_loss = stats[2]
		if np.abs(valid_loss + self.es.min_delta) < np.abs(self.best_valid_loss):
			self.best_ep = ep
			self.best_valid_loss = valid_loss
			# self.best_model = self._net
			self.cpt_patience = 0
		else:
			self.cpt_patience += 1
			if self.cpt_patience >= self.es.patience:
				return True
		return False

	def _update_train_loss(self, ep:int, stats:Tuple[float,float,float,float]) -> bool:
		train_loss = stats[0]
		if np.abs(train_loss + self.es.min_delta) < np.abs(self.best_train_loss):
			self.best_ep = ep
			self.best_train_loss = train_loss
			# self.best_model = self._net
			self.cpt_patience = 0
		else:
			self.cpt_patience += 1
			if self.cpt_patience >= self.es.patience:
				return True
		return False

	def _update_valid_accu(self, ep:int, stats:Tuple[float,float,float,float]) -> bool:
		valid_acc = stats[3]
		if (valid_acc - self.es.min_delta) > self.best_valid_acc:
			self.best_ep = ep
			self.best_valid_acc = valid_acc
			# self.best_model = self._net
			self.cpt_patience = 0
		else:
			self.cpt_patience += 1
			if self.cpt_patience >= self.es.patience:
				return True
		return False

	def _update_train_accu(self, ep:int, stats:Tuple[float,float,float,float]) -> bool:
		train_acc = stats[1]
		if (train_acc - self.es.min_delta) > self.best_train_acc:
			self.best_ep = ep
			self.best_train_acc = train_acc
			# self.best_model = self._net
			self.cpt_patience = 0
		else:
			self.cpt_patience += 1
			if self.cpt_patience >= self.es.patience:
				return True
		return False
