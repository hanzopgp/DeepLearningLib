import numpy as np
import nndyi.core


class LeakyReLU(nndyi.core.Activation):
	ALPHA = 1e-2

	def forward(self, data):
		self._input = data
		self._output = np.where(self._input > 0, self._input, self.ALPHA * self._input)

	def backward(self):
		self._new_delta = np.where(self._input > 0, 1, self.ALPHA) * self._delta


class Identity(nndyi.core.Activation):
	def forward(self, data):
		self._input = self._output = data

	def backward(self):
		self._new_delta = np.ones_like(self._input) * self._delta


class ReLU(nndyi.core.Activation):
	def forward(self, data):
		self._input = data
		self._output = np.maximum(0, self._input)

	def backward(self):
		self._new_delta = np.where(self._input > 0, 1, 0) * self._delta


class Sigmoid(nndyi.core.Activation):
	def forward(self, data):
		self._input = data
		self._output = 1/(1 + np.exp(-data))

	def backward(self):
		sigmoid = 1/(1 + np.exp(-self._input))
		self._new_delta = sigmoid * (1 - sigmoid) * self._delta


class Softmax(nndyi.core.Activation):
	def forward(self, data):
		self._input = data - np.max(data, axis=1, keepdims=True)	# Fixes overflow issue in exp()
		exp_ = np.exp(self._input)
		self._output = exp_ / np.sum(exp_, axis=1).reshape(-1, 1)

	def backward(self):
		_exp = np.exp(self._input)
		soft = _exp / np.sum(_exp, axis=1).reshape(-1, 1)
		self._new_delta = self._delta * (soft*(1-soft))


class Tanh(nndyi.core.Activation):
	def forward(self, data):
		self._input = data
		self._output = np.tanh(self._input)

	def backward(self):
		self._new_delta = 1 - np.tanh(self._input)**2 * self._delta
