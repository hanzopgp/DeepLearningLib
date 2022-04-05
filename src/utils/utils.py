import numpy as np


def one_hot(y, n_class):
	n = y.shape[0]
	onehot = np.zeros((n, n_class))
	onehot[np.arange(n), y] = 1
	return onehot

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
	std_ = (X - min_) / (max_ - min_)
	return std_ * (max_input - min_input) + min_input
