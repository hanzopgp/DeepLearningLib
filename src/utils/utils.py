import numpy as np

def one_hot(y):
	n = y.shape[0]
	onehot = np.zeros((n, len(np.unique(y))))
	onehot[np.arange(n), y] = 1
	return onehot