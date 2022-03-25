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
	## The forward pass is the commputation of output thanks to <input> (and parameters if the module contains parameters)
	## The forward pass can occur on a layer, an activation function or a loss function.
	def forward(self, input):
		raise NotImplementedError()

	## Updates the <_parameters> according to the <_gradient> and the <learning_rate>
	def update_parameters(self, learning_rate):
		raise NotImplementedError()

	## Computes the weights gradient (using the <input> which is the X data input)
	## This is the gradient we will keep in the module in order to update the parameters
	## Since we might use minibatch or batch gradient descent, the gradient has to be added each pass
	## The <delta> argument is the gradient coming from the next module
	## We need to multiply by <delta> after computing our current gradient
	def backward_update_gradient(self, delta):
		raise NotImplementedError()

	## Computes the inputs gradient (using the <input> which is the X data input)
	## This is the gradient that is backpropagated throught the modules in order to compute previous modules gradients
	## The <delta> argument is the delta coming from the next module
	## We need to multiply by <delta> after computing our current gradient
	def backward_delta(self):
		raise NotImplementedError()
