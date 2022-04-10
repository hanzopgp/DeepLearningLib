import numpy as np


DIVIDE_BY_ZERO_EPS = 1e-9
MIN_THRESHOLD = 1e-5
MAX_THRESHOLD = 1e5


class Module():
	def __init__(self):
		self._parameters = None
		self._gradient = None

	def zero_grad(self):
		"""Resets gradient values"""
		raise NotImplementedError()

	def forward(self, data:np.ndarray):
		"""Computes forward pass thanks to `data` (and parameters if the module contains parameters).
		The forward pass can occur on a MLP, an activation layer or a loss layer."""
		raise NotImplementedError()

	def backward(self, truth:np.ndarray, prediction:np.ndarray):
		"""Computes the backward pass"""
		raise NotImplementedError()

	def update_parameters(self, learning_rate:float):
		"""Updates the module's parameters using the computed gradient and the `learning_rate`"""
		raise NotImplementedError()

	def backward_update_gradient(self, delta:np.ndarray):
		"""Computes the weights gradient using the input from the forward pass.
		This is the gradient we will keep in order to update the parameters of the module.
		Since we might use mini-batch or batch gradient descent, the gradient has to be added each pass.
		The `delta` argument is the gradient coming from the module after.
		We need to multiply by `delta` after computing our current gradient"""
		raise NotImplementedError()

	def backward_delta(self):
		"""Computes the inputs gradient using the input from the forward pass.
		This is the gradient back-propagated through the modules in order to compute previous modules' gradients.
		The `delta` argument is the gradient coming from the module after.
		We need to multiply by `delta` after computing our current gradient"""
		raise NotImplementedError()


class Loss(Module):
	def zero_grad(self):
		pass
	
	def backward(self):
		raise NotImplementedError()


class Activation(Module):
	def zero_grad(self):
		pass
	
	def update_parameters(self, learning_rate:float):
		pass
	
	def backward_update_gradient(self, delta:np.ndarray):
		self._delta = delta


class Optimizer():
	def __init__(self, learning_rate:float, decay:float, n_batch:int):
		self._lr = learning_rate
		self._decay = decay
		self._n_batch = n_batch
	
	def step(self, X:np.ndarray, y:np.ndarray, n_epochs:int, verbose:bool, early_stopping):
		raise NotImplementedError()
