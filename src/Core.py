import numpy as np

class Loss():
	def forward(self, y, yhat):
		raise NotImplementedError()

	def backward(self, y, yhat):
		raise NotImplementedError()


class Module():
	def __init__(self):
		self._parameters = None
		self._gradient = None

	def zero_grad(self):
		## Annule gradient
		raise NotImplementedError()

	def forward(self, X):
		## Calcule la raise NotImplementedError()e forward
		raise NotImplementedError()

	def update_parameters(self, gradient_step=1e-3):
		## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
		self._parameters -= gradient_step*self._gradient

	def backward_update_gradient(self, input, delta):
		## Met a jour la valeur du gradient
		raise NotImplementedError()

	def backward_delta(self, input, delta):
		## Calcul la derivee de l'erreur
		raise NotImplementedError()
