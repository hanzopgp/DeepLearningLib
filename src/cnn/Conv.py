from Core import *
from global_imports import *

class Convo1D(Module):
	def __init__(self, kernel_size:int, chan_in:int, chan_out:int, stride:int):
		self._parameters:np.ndarray = (np.random.rand(chan_out, kernel_size, chan_in) * 2 - 1) / np.sqrt(chan_in)	# xavier-init
		self._stride = stride
		self.zero_grad()

	def zero_grad(self):
		self._gradient = np.zeros_like(self._parameters)

	def forward(self, inp:np.ndarray):
		chan_out, ksz, chan_in = self._parameters.shape
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
			for b in range(batch_sz):	# Need to find a way to process batch with numpy rather than loopoig here
				window = inp[b,i:i+ksz,:]
				self._output[b,idx_out,:] = np.sum(self._parameters * window, axis=(1,2))
			idx_out += 1

	# def update_parameters(self, learning_rate):
	# 	self._parameters -= (learning_rate * self._gradient)

	# def backward_update_gradient(self, delta):
	# 	self._delta = delta
	# 	self._gradient += self._input.T @ self._delta

	# def backward_delta(self):
	# 	self._new_delta = self._delta @ self._parameters.T
