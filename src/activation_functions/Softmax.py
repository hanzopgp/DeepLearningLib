from Core import *

# Source: https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/

class Softmax(Module):
	def forward(self, input):
		return np.exp(input) / np.sum(np.exp(input))

	def update_parameters(self, learning_rate=1e-3):
		self._parameters -= learning_rate*self._gradient

	def backward_update_gradient(self, input, delta):
		input_size = input.shape[0]
		input_vector = input.reshape(input_size, 1)
		input_matrix = np.tile(input_vector, input_size) # np.tile repeats the <input_vector> array <input_size> times
		self._gradient = np.diag(input) - (input_matrix * input_matrix.T) + delta
		return self._gradient

	def backward_delta(self, input, delta):
		raise NotImplementedError()

# Quotient rule for derivatives
# Derivatives are 0 if i!=j and exp(x_j) else. (i and j are the index of the classes)
# Evaluate the quotient rule for both cases
# Find a way to implement it in python
# We can do that by using the jacobian
# We just need the diagonal so that's why we use np.diag()
# 