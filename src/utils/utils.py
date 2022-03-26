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