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

	## Updates the parameters according to the gradient and the learning rate
	def update_parameters(self, learning_rate=1e-3):
		self._parameters -= learning_rate*self._gradient

	## Computes the weights gradient (thanks to the loss, which is the input in our arguments)
	## This is the gradient we will keep in the module in order to update the parameters
	## Since we might use minibatch or batch gradient descent, the gradient has to be added each pass 
	## The delta argument is the delta coming from the next module
	def backward_update_gradient(self, input, delta):
		raise NotImplementedError()

	## Computes the input gradient (thanks to the loss, which is the input in our arguments)
	## This is the gradient that is backpropagated throught the modules in order to compute previous modules gradients
	## The delta argument is the delta coming from the next module
	def backward_delta(self, input, delta):
		raise NotImplementedError()
