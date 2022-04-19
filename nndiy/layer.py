import numpy as np
import nndiy.core
from numba import njit


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
		self._grad_input = self._delta @ self._W.T

	def zero_grad(self):
		self._grad_W = np.zeros_like(self._W)
		self._grad_b = np.zeros_like(self._b)

	def backward_update_gradient(self, delta):
		self._delta = delta
		self._grad_W += self._input.T @ delta
		self._grad_b += np.sum(delta, axis=0)


class Convo1D(nndiy.core.Module):
	def __init__(self, kernel_size:int, chan_in:int, chan_out:int, stride=1, regularization=1e-9):
		self._ksz = kernel_size
		self._chan_in = chan_in
		self._chan_out = chan_out
		self._lambda = regularization

		self._W:np.ndarray = (np.random.rand(chan_out, kernel_size, chan_in) * 2 - 1) / np.sqrt(chan_in)	# xavier-init
		self._b = 0
		self._stride = stride

		self._grad_W = np.zeros_like(self._W)
		self._grad_b = 0

	def zero_grad(self):
		pass

	@staticmethod
	@njit
	def forward_nb(W, inp, stride):
		chan_out, ksz, _ = W.shape
		batch_sz, length, _ = inp.shape
		d_out = (length - ksz) // stride + 1
		output = np.zeros((batch_sz, d_out, chan_out))
		idx_out = 0
		for i in range(0, length, stride):
			if idx_out == d_out:
				break
			for b in range(batch_sz):
				window = inp[b, i:i+ksz, :]
				for c in range(chan_out):
					kernel = W[c, :, :]
					output[b, idx_out, c] = np.sum(kernel * window)
				# output[b, idx_out, :] = np.sum(W * window, axis=(1,2))
			idx_out += 1
		return output

	def forward(self, inp:np.ndarray):
		# input : (batchsz, length, chan_in)
		# output: (batchsz, d_out, chan_out)
		assert inp.shape[2] == self._chan_in
		self._batch_sz, self._length, _ = inp.shape

		self._input = inp
		self._output = self.forward_nb(self._W, inp, self._stride)
		# d_out = (self._length - self._ksz) // self._stride + 1
		# self._output = np.zeros((self._batch_sz, d_out, self._chan_out))
		
		# # Parallelizing may be possible here
		# idx_out = 0
		# for i in range(0, self._length, self._stride):
		# 	if idx_out == d_out:
		# 		break
		# 	for b in range(self._batch_sz):	# Need to find a way to process batch with numpy rather than looping here
		# 		window = inp[b, i:i+self._ksz, :]
		# 		self._output[b, idx_out, :] = np.sum(self._W * window, axis=(1,2))
		# 	idx_out += 1

	@staticmethod
	@njit
	def backward_nb(W, inp, delta, stride):
		chan_out, ksz, _ = W.shape
		batch_sz, length, _ = inp.shape
		d_out = delta.shape[1]

		grad_input = np.zeros_like(inp)
		grad_W = np.zeros_like(W)
		idx_out = 0
		for i in range(0, length, stride):
			if idx_out == d_out:
				break
			for c in range(chan_out):
				for b in range(batch_sz):
					kernel = W[c, :, :]	# ksz, chan_in
					grad_input[b, i:i+ksz, :] += kernel[::-1, :] * delta[b, idx_out, c]
					grad_W[c, :, :] += inp[b, i:i+ksz, :] * delta[b, idx_out, c]
			idx_out += 1
		return grad_input, grad_W

	def backward(self):
		# output: (batch_sz, length, chan_in)
		self._grad_input, self._grad_W = self.backward_nb(self._W, self._input, self._delta, self._stride)
		# self._grad_input = np.zeros_like(self._input)
		# d_out = self._output.shape[1]
		# idx_out = 0
		# for i in range(0, self._length, self._stride):
		# 	if idx_out == d_out:
		# 		break
		# 	for c in range(self._chan_out):
		# 		for b in range(self._batch_sz):
		# 			kernel = self._W[c, :, :]	# ksz, chan_in
		# 			self._grad_input[b, i:i+self._ksz, :] += kernel * self._delta[b, idx_out, c]
		# 			self._grad_W[c, :, :] += self._input[b, i:i+self._ksz, :] * self._delta[b, idx_out, c]
		# 	idx_out += 1
					

	def backward_update_gradient(self, delta):
		# delta: (batch_sz, d_out, chan_out)
		self._delta = delta


class MaxPool1D(nndiy.core.Module):
	def __init__(self, kernel_size:int, stride=1):
		self._ksz = kernel_size
		self._stride = stride
		self._saved_argmax = None

	def zero_grad(self):
		pass

	@staticmethod
	@njit
	def forward_nb(inp, ksz, stride):
		batch_sz, length, chan = inp.shape
		d_out = int(np.floor((length - ksz) / stride)) + 1
		
		output = np.zeros((batch_sz, d_out, chan))
		saved_argmax = np.zeros_like(output, dtype=np.int32)

		idx_out = 0
		for i in range(0, length, stride):
			if idx_out == d_out:
				break
			for b in range(batch_sz):
				for c in range(chan):
					window = inp[b, i:i+ksz, c]
					output[b, idx_out, c] = np.max(window)
					saved_argmax[b, idx_out, c] = np.argmax(window)
			idx_out += 1
		return output, saved_argmax

	def forward(self, inp:np.ndarray):
		self._length = inp.shape[1]
		self._output, self._saved_argmax = self.forward_nb(inp, self._ksz, self._stride)
		# if self._mask is None:
		# 	self._mask = np.zeros(inp.shape, dtype=np.uint8)
		# batch_sz, length, chan = inp.shape
		# d_out = int(np.floor((length - self._ksz) / self._stride)) + 1
		# self._input = inp
		# self._output = np.zeros((batch_sz, d_out, chan))

		# idx_out = 0
		# for i in range(0, length, self._stride):
		# 	if idx_out == d_out:
		# 		break
		# 	for b in range(batch_sz):	# Need to find a way to process batch with numpy rather than loopoig here
		# 		window = inp[b,i:i+self._ksz,:]
		# 		self._output[b,idx_out,:] = np.max(window, axis=0)
		# 		argmax = np.argmax(window, axis=0)
		# 		self._mask[b,argmax+i,np.arange(argmax.shape[0])] = 1
		# 	idx_out += 1

	@staticmethod
	@njit
	def backward_nb(delta, length, stride, saved_argmax):
		batch_sz, d_out, chan = delta.shape
		grad_input = np.zeros((batch_sz, length, chan))

		for b in range(batch_sz):
			for c in range(chan):
				for i in range(d_out):
					grad_input[b, i*stride + saved_argmax[b, i, c], c] = delta[b, i, c]
		return grad_input

	def backward(self):
		self._grad_input = self.backward_nb(self._delta, self._length, self._stride, self._saved_argmax)

	def backward_update_gradient(self, delta):
		self._delta = delta


class Flatten(nndiy.core.Module):
	def forward(self, inp:np.ndarray):
		self._batch_sz, self._length, self._chan = inp.shape
		self._input = inp
		self._output = inp.reshape(self._batch_sz, self._length * self._chan, order='C')

	def backward(self):
		self._grad_input = self._delta.reshape(self._batch_sz, self._length, self._chan, order='C')

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
		self._grad_input = self._delta * self._mask
