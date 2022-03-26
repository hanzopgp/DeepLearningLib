from Core import *

# Source: https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/

class Softmax(Module):
	def zero_grad(self):
		pass

	def update_parameters(self, learning_rate):
		pass
	
	def forward(self, input):
		self._input = input
		exp_ = np.exp(self._input)
		self._output = exp_ / np.sum(exp_)
		assert(self._input.shape == self._output.shape)

	def backward_update_gradient(self, delta):
		assert(delta.shape == self._output.shape)
		self._delta = delta

	def backward_delta(self):
		input_size = self._input.shape[0]
		input_vector = self._input.reshape(input_size, 1)
		## np.tile repeats the <input_vector> array <input_size> times
		input_matrix = np.tile(input_vector, input_size) 
		gradient = np.diag(self._input) - (input_matrix * input_matrix.T)
		self._new_delta = gradient * self._delta

## Quotient rule for derivatives
## Derivatives are 0 if i!=j and exp(x_j) else. (i and j are the index of the classes)
## Evaluate the quotient rule for both cases
## Find a way to implement it in python
## We can do that by using the jacobian
## We just need the diagonal so that's why we use np.diag()
