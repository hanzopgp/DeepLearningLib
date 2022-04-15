import numpy as np
import nndiy.core


class Convo1D(nndiy.core.Module):
	def __init__(self, kernel_size:int, chan_in:int, chan_out:int, stride=1):
		self._W:np.ndarray = (np.random.rand(chan_out, kernel_size, chan_in) * 2 - 1) / np.sqrt(chan_in)	# xavier-init
		self._b:np.ndarray = np.zeros(chan_out)
		self._stride = stride
		self.zero_grad()

	def zero_grad(self):
		self._grad_W = np.zeros_like(self._W)
		self._grad_b = np.zeros_like(self._b)

	def forward(self, inp:np.ndarray):
		# input : (batchsz, length, chan_in)
		# output: (batchsz, d_out, chan_out)
		chan_out, ksz, chan_in = self._W.shape
		batch_sz = inp.shape[0]
		length = inp.shape[1]
		assert inp.shape[2] == chan_in

		d_out = int(np.floor((length - ksz) / self._stride)) + 1
		self._input = inp
		self._output = np.zeros((batch_sz, d_out, chan_out))

		# Parallelizing may be possible here
		idx_out = 0
		for i in range(0, length, self._stride):
			if idx_out == d_out:
				break
			for b in range(batch_sz):	# Need to find a way to process batch with numpy rather than looping here
				window = inp[b,i:i+ksz,:]
				self._output[b,idx_out,:] = np.sum(self._W * window, axis=(1,2)) + self._b
			idx_out += 1

	def backward(self):
		# output: (batch_sz, length, chan_in)
		batch_sz = self._input.shape[0]
		self._new_delta = np.zeros_like(self._input)
		for b in range(batch_sz):
			self._new_delta[b] = self._delta[b] @ self._W.T

	def backward_update_gradient(self, delta):
		# delta: (batch_sz, d_out, chan_out)
		batch_sz = delta.shape[0]
		self._delta = delta
		# print("$"*30)
		# print(self._input.shape, delta.shape)
		# print("$"*30)
		for b in range(batch_sz):
			self._grad_W[b] += self._input[b].T @ delta[b]


class MaxPool1D(nndiy.core.Module):
	def __init__(self, kernel_size:int, stride=1):
		self._ksz = kernel_size
		self._stride = stride
		self._mask = None

	def zero_grad(self):
		self._mask = np.zeros_like(self._mask)

	def forward(self, inp:np.ndarray):
		if self._mask is None:
			self._mask = np.zeros(inp.shape, dtype=np.uint8)
		batch_sz, length, chan = inp.shape
		d_out = int(np.floor((length - self._ksz) / self._stride)) + 1
		self._input = inp
		self._output = np.zeros((batch_sz, d_out, chan))

		idx_out = 0
		for i in range(0, length, self._stride):
			if idx_out == d_out:
				break
			for b in range(batch_sz):	# Need to find a way to process batch with numpy rather than loopoig here
				window = inp[b,i:i+self._ksz,:]
				self._output[b,idx_out,:] = np.max(window, axis=0)
				argmax = np.argmax(window, axis=0)
				self._mask[b,argmax+i,np.arange(argmax.shape[0])] = 1
			idx_out += 1

	def backward(self):
		# batch_sz = self._input.shape[0]
		# self._new_delta = np.zeros_like(self._input)
		# for b in range(batch_sz):
		# 	self._new_delta[b] = self._mask * self._delta[b]
		self._new_delta = self._mask * np.repeat(self._delta, self._stride, 1)

	def backward_update_gradient(self, delta):
		self._delta = delta


class Flatten(nndiy.core.Module):
	def forward(self, inp:np.ndarray):
		batch_sz, length, chan = inp.shape
		self._input = inp
		self._output = inp.reshape(batch_sz, -1)

	def backward(self):
		self._new_delta = self._delta.reshape(self._input.shape)

	def backward_update_gradient(self, delta):
		self._delta = delta
