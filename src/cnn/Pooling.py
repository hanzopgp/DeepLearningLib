from Core import *
from global_imports import *

class MaxPool1D(Module):
	def __init__(self, kernel_size:int, stride:int):
		self._ksz = kernel_size
		self._stride = stride

	def zero_grad(self):
		self._gradient = np.zeros_like(self._gradient)

	def forward(self, inp:np.ndarray):
		if self._gradient is None:
			self._gradient = np.zeros_like(inp)
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
				self._gradient[b,argmax+i,np.arange(argmax.shape[0])] = 1
			idx_out += 1


class Flatten(Module):
	def forward(self, inp:np.ndarray):
		batch_sz, length, chan = inp.shape
		self._input = inp
		self._output = inp.reshape((batch_sz, length * chan))
