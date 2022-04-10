import numpy as np
from core import Loss, MIN_THRESHOLD, MAX_THRESHOLD, DIVIDE_BY_ZERO_EPS
from utils import one_hot


class BinaryCrossEntropy(Loss):
	def forward(self, truth, prediction):
		self._y = truth
		self._yhat = np.where(prediction < MIN_THRESHOLD, MIN_THRESHOLD, prediction)
		self._yhat = np.where(self._yhat > MAX_THRESHOLD, MAX_THRESHOLD, self._yhat)
		self._output = (1 - self._y) * np.log(1 - self._yhat + DIVIDE_BY_ZERO_EPS) \
			- self._y * np.log(self._yhat + DIVIDE_BY_ZERO_EPS)

	def backward(self):
		self._new_delta = ((1 - self._y) / (1 - self._yhat + DIVIDE_BY_ZERO_EPS)) \
			- (self._y / self._yhat + DIVIDE_BY_ZERO_EPS)


class SparseBinaryCrossEntropy(BinaryCrossEntropy):
	def forward(self, truth, prediction):
		super().forward(one_hot(truth, prediction.shape[1]), prediction)


class CategoricalCrossEntropy(Loss):
	def forward(self, truth, prediction):
		self._y = truth
		self._yhat = prediction
		self._output = 1 - np.sum(self._yhat * self._y, axis=1)

	def backward(self):
		self._new_delta = self._yhat - self._y


class SparseCategoricalCrossEntropy(CategoricalCrossEntropy):
	def forward(self, truth, prediction):
		super().forward(one_hot(truth, prediction.shape[1]), prediction)


class SparseCategoricalCrossEntropySoftmax(Loss):
	def forward(self, truth, prediction):
		self._y = one_hot(truth, prediction.shape[1])
		self._yhat = np.where(prediction < MIN_THRESHOLD, MIN_THRESHOLD, prediction)
		self._yhat = np.where(self._yhat > MAX_THRESHOLD, MAX_THRESHOLD, self._yhat)
		self._output = np.log(np.sum(np.exp(self._yhat), axis=1)) \
			- np.sum(self._y * self._yhat, axis=1)

	def backward(self):
		_exp = np.exp(self._yhat)
		self._new_delta = _exp / (np.sum(_exp, axis=1).reshape((-1, 1)) + DIVIDE_BY_ZERO_EPS) - self._y


class MeanAbsoluteError(Loss):
	def forward(self, truth, prediction):
		self._y = truth
		self._yhat = prediction
		self._output = np.sum(np.abs(self._y - self._yhat), axis=1)

	def backward(self):
		self._new_delta = np.where(self._yhat > self._y, 1, -1)


class MeanSquaredError(Loss):
	def forward(self, truth, prediction):
		self._y = truth
		self._yhat = prediction
		self._output = np.sum((self._y - self._yhat) ** 2, axis=1)

	def backward(self):
		self._new_delta = -2 * (self._y - self._yhat)


class RootMeanSquaredError(Loss):
	def forward(self, truth, prediction):
		self._y = truth
		self._yhat = prediction
		self._output = np.sum(np.sqrt((self._y - self._yhat) ** 2), axis=1)

	def backward(self):
		mse = (self._y - self._yhat) ** 2
		d_mse = - 2 * (self._y - self._yhat)
		self._new_delta = 1/(2*np.sqrt(mse)) * (d_mse)
