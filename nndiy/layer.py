import numpy as np
import nndiy.core


LINEAR_INIT_ARGS = (None, 'xavier')
PARAMS_INIT_FACTOR = 1e-3
OVERFLOW_THRESHOLD = 1e6


class Linear(nndiy.core.Module):
	"""A layer consist of multiple linear neurons, each with its own parameters (weight and bias)"""
	def __init__(self, size_in:int, size_out:int, init='xavier', regularization=1e-9):
		assert size_in > 0
		assert size_out > 0
		assert init in LINEAR_INIT_ARGS
		
		if init is None:
			self._W = np.random.rand(size_in, size_out) * PARAMS_INIT_FACTOR
		else:	# xavier initialization
			self._W = (np.random.rand(size_in, size_out)*2 - 1) / np.sqrt(size_out)
		self._b = np.zeros(size_out)
		self._lambda = regularization
		self.zero_grad()

	def forward(self, data):
		self._input = data
		self._output = data @ self._W + self._b
		# Clip output to avoid overflowing in activation layer
		self._output = np.where(self._output > OVERFLOW_THRESHOLD, OVERFLOW_THRESHOLD, self._output)
		self._output = np.where(self._output < -OVERFLOW_THRESHOLD, -OVERFLOW_THRESHOLD, self._output)
	
	def backward(self):
		self._new_delta = self._delta @ self._W.T

	def zero_grad(self):
		self._grad_W = np.zeros_like(self._W)
		self._grad_b = np.zeros_like(self._b)

	def backward_update_gradient(self, delta):
		self._delta = delta
		self._grad_W += self._input.T @ delta
		self._grad_b += np.sum(delta, axis=0)

class Convo1D(nndiy.core.Module):
	def __init__(self, kernel_size:int, chan_in:int, chan_out:int, stride=1):
		self._W:np.ndarray = (np.random.rand(chan_out, kernel_size, chan_in) * 2 - 1) / np.sqrt(chan_in) # xavier-init
		self._b:np.ndarray = np.zeros(chan_out)
		self._ksz = kernel_size
		self._stride = stride
		self._chan_in = chan_in
		self._chan_out = chan_out
		self.zero_grad()

	def zero_grad(self):
		self._grad_W = np.zeros_like(self._W)
		self._grad_b = np.zeros_like(self._b)

	def forward(self, inp:np.ndarray):
		self._input = inp
		self._batch_size = inp.shape[0]
		tmp = []
		for x in self._input: # For each element our batch
			tmp_x = []
			for filter in self._W: # For each filter of our layer
				for i in range(0, x.shape[0], self._stride): # For each dimension of our input
					x_in = x[i:i+self._stride] # We take the part we want to convolve (x_in)
					tmp_x.append(sum(x_in*filter)) # We do the weighted sum which gives us tmp_x
			tmp_x = np.array(tmp_x).reshape((self._W.shape[0], -1))
			tmp.append(tmp_x.T) # We add the result of our filter on x_in after reshaping it
		self._output = np.array(tmp)
		# Clip output to avoid overflowing in activation layer
		self._output = np.where(self._output > OVERFLOW_THRESHOLD, OVERFLOW_THRESHOLD, self._output)
		self._output = np.where(self._output < -OVERFLOW_THRESHOLD, -OVERFLOW_THRESHOLD, self._output)

	def backward(self):
		tmp = []
		for b in range(self._batch_size): # For each element of our batch
			tmp_x = []
			for f in range(self._W.shape[0]): # For each filter of our layer
				for col_d in self._delta[b, :, f]: # Here we iterate on the columns of the deltas 
					tmp_x.extend(np.multiply(self._W[f], col_d)) # Then we multiply is by each filters to get the gradient
			tmp_x = np.array(tmp_x).reshape((self._chan_out, -1)).T
			tmp.append(tmp_x) # Like in forward pass we get several gradients corresponding to each filters
		self._new_delta = np.array(tmp)

	def backward_update_gradient(self, delta):
		self._delta = delta
		for b in range(self._batch_size): # For each element of our batch
			for f in range(self._W.shape[0]): # For each filter of our layer
				# We need to do some reshape on our inputs for further computation
				tmp = self._input[b].reshape((self._input.shape[1] // self._stride, -1))
				# As in linear layer we get the gradient with respect to the input
				# by multiplying the input with the right delta values
				self._grad_W[f] += tmp.T @ delta[b, :, f]
				# The bias gradient is the sum of the residuals
				self._grad_b[f] += np.sum(delta[b, :, f])


class MaxPool1D(nndiy.core.Module):
	def __init__(self, kernel_size:int, stride=1):
		self._ksz = kernel_size
		self._stride = stride

	def forward(self, inp:np.ndarray):
		self._input = inp
		self._batch_size = inp.shape[0]
		self._channels_size = inp.shape[2]
		tmp = [] # This array contains the would output
		for b in range(self._batch_size): # For each element of the batch
			tmp_e = [] # This array contains the array of max_values for one element
			dim = self._input[b].shape[0]
			for j in range(self._input[b].shape[1]): # For each dimension of the element
				tmp_max = [] # This array contains our max() values of one window
				for i in range(0, dim, self._stride): # For each rows
					if i + self._ksz <= dim: # If our sliding window is in range
						max = (self._input[b, i:i+self._ksz, j]).max() # We compute the maximum
						tmp_max.append(max)
				tmp_e.append(tmp_max)
			tmp.append(np.array(tmp_e).T)
		self._output = np.array(tmp)

	def backward(self):
		tmp = np.zeros_like(self._input)
		for x in range(self._batch_size): # For each element
			for c in range(self._channels_size): # For each channels
				for i in range(self._delta.shape[1]): # For each index
					# This is the array we are looking at according to the kernel size and stride
					arr = self._input[x, i*self._stride:i*self._stride+self._ksz, c]
					# This is the maximum value of the array we are looking at
					max = (self._input[x, i*self._stride:i*self._stride+self._ksz, c]).max()
					# We get the index of maximum values
					max_index = np.where(arr==max)[0][0]
					# And we backward the delta from previous layer only where the values are max
					tmp[x, i*self._stride+max_index, c]=self._delta[x,i,c]
		self._new_delta = tmp

	def backward_update_gradient(self, delta):
		self._delta = delta


class Flatten(nndiy.core.Module):
	def forward(self, inp:np.ndarray):
		batch_sz, _, _ = inp.shape
		self._input = inp
		self._output = inp.reshape(batch_sz, -1)

	def backward(self):
		self._new_delta = self._delta.reshape(self._input.shape)

	def backward_update_gradient(self, delta):
		self._delta = delta


class Dropout(nndiy.core.Module):
	"""Drop-out layer with a fixed drop-out rate"""
	def __init__(self, rate=0.2):
		self._rate = rate

	def forward(self, data):
		self._input = data
		# Bernouilli mask scaled by 1/rate
		self._mask = np.random.binomial(1, self._rate, size=self._input.shape) / self._rate
		# To randomly disable the inputs we can multipy elementwise the input with the mask
		self._output = self._input * self._mask

	def backward_update_gradient(self, delta):
		self._delta = delta

	def backward(self):
		# Here we only back propagate the delta for inputs which weren't disabled during forward pass,
		# so we can just multiply elementwise the delta with our mask again
		self._new_delta = self._delta * self._mask
