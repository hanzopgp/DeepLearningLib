import numpy as np
from dataclasses import dataclass
from . import Sequential
from optimizer import EarlyStopping


DIVIDE_BY_ZERO_EPS = 1e-9
MIN_THRESHOLD = 1e-5
MAX_THRESHOLD = 1e5


class Module():
	def forward(self, data:np.ndarray):
		"""Computes forward pass thanks to `data` (and parameters if the module contains parameters).
		The forward pass can occur on a MLP, an activation layer or a loss layer."""
		raise NotImplementedError()

	def backward(self):
		"""Computes the backward pass using the input from the forward pass.
		This is the gradient back-propagated through the modules in order to
		compute previous modules' gradients."""
		raise NotImplementedError()

	def zero_grad(self):
		"""Resets gradient values"""
		raise NotImplementedError()

	def backward_update_gradient(self, delta:np.ndarray):
		"""Computes the weights gradient using the input from the forward pass.
		This is the gradient we will keep in order to update the parameters of the module.
		Since we might use mini-batch or batch gradient descent, the gradient has to be added each pass.
		The `delta` argument is the gradient coming from the module after.
		We need to multiply by `delta` after computing our current gradient"""
		raise NotImplementedError()
	

class Activation(Module):
	def backward_update_gradient(self, delta:np.ndarray):
		self._delta = delta


class Loss(Module):
	def forward(self, truth:np.ndarray, prediction:np.ndarray):
		raise NotImplementedError()


@dataclass
class EarlyStoppingHandler():
	es_metric:str
	es_min_delta:float
	es_patience:int
	best_model:Sequential
	best_cpt_epoch:int = 0
	best_valid_loss:float = np.inf
	best_train_loss:float = np.inf
	best_valid_acc:float = 0.
	best_train_acc:float = 0.
	cpt_patience:int = 0

	def update(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float) -> bool:
		pass


class Optimizer():
	def __init__(self, network:Sequential, learning_rate:float, decay:float, n_batch:int):
		self._seq = network
		self._lr = learning_rate
		self._decay = decay
		self._n_batch = n_batch
	
	def step(self, X:np.ndarray, y:np.ndarray, n_epochs:int, verbose:bool, early_stopping:EarlyStopping):
		raise NotImplementedError()

	def _update_layer_params(self, layer:Module):
		raise NotImplementedError()

	def _make_es_handler(self, early_stopping:EarlyStopping):
		if early_stopping is not None:
			self._es = EarlyStoppingHandler(*early_stopping, self._seq)

	def _show_updates(self, epoch:int, train_loss: float, train_acc:float, valid_loss:float, valid_acc:float):
		epoch = "%04d" % (epoch)
		train_loss = ("%10f" % (train_loss)) if train_loss is not None else None
		train_acc = ("%10f" % (train_acc)) if train_acc is not None else None
		valid_loss = ("%10f" % (valid_loss)) if valid_loss is not None else None
		valid_acc = ("%10f" % (valid_acc)) if valid_acc is not None else None
		print(f"epoch {epoch},",
			f"train_loss {train_loss},",
			f"train_accuracy {train_acc},",
			f"validation_loss {valid_loss},",
			f"validation_accuracy {valid_acc},",
			f"learning_rate {self._lr}")
