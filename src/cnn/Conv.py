from Core import *
from global_imports import *

class Convo1D(Module):
	def __init__(self, kernel_size, chan_in, chan_out, stride):
		self._parameters = (np.random.rand(kernel_size, chan_in) * 2 - 1) / np.sqrt(chan_in)	# xavier-init
		self._stride = stride
