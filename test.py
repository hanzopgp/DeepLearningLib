import numpy as np
from nndiy import Sequential
from nndiy.layer import Linear, Convo1D, MaxPool1D, Flatten

USPS_TRAIN = "data/USPS_train.txt"
USPS_TEST = "data/USPS_test.txt"

def load_usps(fn):
	with open(fn,"r") as f:
		f.readline()
		data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
	tmp=np.array(data)
	return tmp[:,1:],tmp[:,0].astype(int)


if __name__ == '__main__':
	np.random.seed(42)
	X_train, y_train = load_usps(USPS_TRAIN)
	X_test, y_test = load_usps(USPS_TEST)
	X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]
	# y_train, y_test = y_train[..., np.newaxis], y_test[..., np.newaxis]

	model = Sequential()
	# model.add(Convo1D(3, 1, 32), activation="identity")
	# model.add(MaxPool1D(2, 2), activation="identity")
	# model.add(Flatten(), activation="identity")
	model._net = [
		Convo1D(3, 1, 32),
		MaxPool1D(2,2),
		Flatten()
	]
	model.add(Linear(4064, 100), activation="relu")
	model.add(Linear(100, 10), activation="softmax")
	model.compile(
		loss="sparse_categorical_crossentropy",
		optimizer="sgd",
		learning_rate=1e-3,
		metric="accuracy"
	)

	model.fit(X_train, y_train, X_test, y_test, n_epochs=100)
