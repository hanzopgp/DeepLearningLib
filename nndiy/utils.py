import numpy as np
from nndiy.core import DIVIDE_BY_ZERO_EPS


def one_hot(y, n_class):
	y_copy = y.reshape(-1)
	res = np.zeros((y.size, n_class), dtype=np.uint8)
	res[np.arange(y.size), y_copy] = 1
	return res

def split_data(X, y, train_split=0.2, shuffle=True):
	assert(X.shape[0] == y.shape[0])
	if shuffle:
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		X, y = X[idx], y[idx]
	split = int(X.shape[0]*train_split)
	return X[:split], X[split:], y[:split], y[split:]

def split_X(X, train_split=0.2, shuffle=True):
	if shuffle:
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		X = X[idx]
	split = int(X.shape[0]*train_split)
	return X[:split], X[split:]

def min_max_scaler(X, min_input, max_input):
	max_ = X.max(axis=0)
	min_ = X.min(axis=0)
	std_ = (X - min_) / (max_ - min_ + DIVIDE_BY_ZERO_EPS)
	return std_ * (max_input - min_input) + min_input
