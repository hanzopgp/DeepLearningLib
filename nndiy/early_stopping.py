import numpy as np
from collections import namedtuple
from typing import Any
from nndiy import Sequential

ES_METRICS = ('valid_loss', 'train_loss', 'valid_accuracy', 'train_accuracy')


EarlyStopping = namedtuple("EarlyStopping", ('metric', 'min_delta', 'patience'))


class EarlyStoppingHandler():
	def __init__(self, es:EarlyStopping, model:Sequential):
		assert es.metric in ES_METRICS

		if es.metric == 'valid_loss':
			self.handle = self._update_valid_loss
		elif es.metric == 'train_loss':
			self.handle = self._update_train_loss
		elif es.metric == 'valid_accuracy':
			self.handle = self._update_valid_accu
		else:
			self.handle = self._update_train_accu

		self.es_metric:str = es.metric
		self.es_min_delta:float = es.min_delta
		self.es_patience:int = es.patience
		self.best_model = model

		self.best_cpt_epoch = 0
		self.best_valid_loss = np.inf
		self.best_train_loss = np.inf
		self.best_valid_acc = 0.
		self.best_train_acc = 0.
		self.cpt_patience = 0

	def _update_valid_loss(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float) -> bool:
		pass

	def _update_train_loss(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float) -> bool:
		pass

	def _update_valid_accu(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float) -> bool:
		pass

	def _update_train_accu(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float) -> bool:
		pass
