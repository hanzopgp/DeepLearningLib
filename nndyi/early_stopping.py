import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from typing import Any
# from nndyi import Sequential


EarlyStopping = namedtuple("EarlyStopping", ('metric', 'min_delta', 'patience'))


@dataclass
class EarlyStoppingHandler():
	es_metric:str
	es_min_delta:float
	es_patience:int
	best_model:Any
	best_cpt_epoch:int = 0
	best_valid_loss:float = np.inf
	best_train_loss:float = np.inf
	best_valid_acc:float = 0.
	best_train_acc:float = 0.
	cpt_patience:int = 0

	def update(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float) -> bool:
		pass
