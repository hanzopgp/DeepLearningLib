import numpy as np
import nndiy.core
from nndiy.core import DIVIDE_BY_ZERO_EPS, MIN_THRESHOLD, MAX_THRESHOLD
from nndiy.utils import one_hot


class BinaryCrossEntropy(nndiy.core.Loss):
	def forward(self, y, yhat):
		self._y = y
		self._yhat = np.where(yhat < MIN_THRESHOLD, MIN_THRESHOLD, yhat)
		self._yhat = np.where(self._yhat > MAX_THRESHOLD, MAX_THRESHOLD, self._yhat)
		self._output = (1 - self._y) * np.log(1 - self._yhat + DIVIDE_BY_ZERO_EPS) \
			- self._y * np.log(self._yhat + DIVIDE_BY_ZERO_EPS)

	def backward(self):
		self._grad_output = ((1 - self._y) / (1 - self._yhat + DIVIDE_BY_ZERO_EPS)) \
			- (self._y / self._yhat + DIVIDE_BY_ZERO_EPS)


class SparseBinaryCrossEntropy(BinaryCrossEntropy):
	def forward(self, y, yhat):
		super().forward(one_hot(y, yhat.shape[1]), yhat)


class CategoricalCrossEntropy(nndiy.core.Loss):
	def forward(self, y, yhat):
		self._y = y
		self._yhat = yhat
		self._output = 1 - np.sum(self._yhat * self._y, axis=1)

	def backward(self):
		self._grad_output = self._yhat - self._y


class SparseCategoricalCrossEntropy(CategoricalCrossEntropy):
	def forward(self, y, yhat):
		super().forward(one_hot(y, yhat.shape[1]), yhat)


class SparseCategoricalCrossEntropySoftmax(nndiy.core.Loss):
	def forward(self, y, yhat):
		self._y = one_hot(y, yhat.shape[1])
		self._yhat = np.where(yhat < MIN_THRESHOLD, MIN_THRESHOLD, yhat)
		self._yhat = np.where(self._yhat > MAX_THRESHOLD, MAX_THRESHOLD, self._yhat)
		self._output = np.log(np.sum(np.exp(self._yhat), axis=1)) \
			- np.sum(self._y * self._yhat, axis=1)

	def backward(self):
		_exp = np.exp(self._yhat)
		self._grad_output = _exp / (np.sum(_exp, axis=1).reshape((-1, 1)) + DIVIDE_BY_ZERO_EPS) - self._y


class MeanAbsoluteError(nndiy.core.Loss):
	def forward(self, y, yhat):
		self._y = y
		self._yhat = yhat
		self._output = np.sum(np.abs(self._y - self._yhat), axis=1)

	def backward(self):
		self._grad_output = np.where(self._yhat > self._y, 1, -1)


class MeanSquaredError(nndiy.core.Loss):
	def forward(self, y, yhat):
		self._y = y
		self._yhat = yhat
		self._output = np.sum((self._y - self._yhat) ** 2, axis=1)

	def backward(self):
		self._grad_output = -2 * (self._y - self._yhat)


class RootMeanSquaredError(nndiy.core.Loss):
	def forward(self, y, yhat):
		self._y = y
		self._yhat = yhat
		self._output = np.sum(np.sqrt((self._y - self._yhat) ** 2), axis=1)

	def backward(self):
		mse = (self._y - self._yhat) ** 2
		d_mse = - 2 * (self._y - self._yhat)
		self._grad_output = 1/(2*np.sqrt(mse)) * (d_mse)
