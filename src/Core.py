import numpy as np

class Loss():
	## Computes forward pass
	def forward(self, y, yhat):
		raise NotImplementedError()

	## Computes backward pass
	def backward(self, y, yhat):
		raise NotImplementedError()

class Module():
	def __init__(self):
		self._parameters = None
		self._gradient = None

	## Resets gradient value
	def zero_grad(self):
		raise NotImplementedError()

	## Computes forward pass
	def forward(self, X):
		raise NotImplementedError()

	## Update the parameters according to the gradient and the learning rate
	def update_parameters(self, learning_rate=1e-3):
		self._parameters -= learning_rate*self._gradient

	## Computes and adds the gradient (adding in case we use minibatch/batch gradient descent)
	def backward_update_gradient(self, input, delta):
		raise NotImplementedError()

	## ???
	def backward_delta(self, input, delta):
		## Calcul la derivee de l'erreur
		raise NotImplementedError()
