import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from nndiy import Sequential
from nndiy.layer import Linear, Convo1D, MaxPool1D, Flatten
from nndiy.early_stopping import EarlyStopping
from nndiy.utils import min_max_scaler, split_data

np.random.seed(42)

label_name_digits_mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def execute_classification_model(X, y, X_test, y_test, label_name, latent=False):
	width = X.shape[1]
	height = X.shape[2]
	## Normalizing our data
	X = min_max_scaler(X, 0, 1)
	X_test = min_max_scaler(X_test, 0, 1)

	X = X.reshape(X.shape[0], -1)[:,:, np.newaxis]
	X_test = X.reshape(X_test.shape[0], -1)[:,:, np.newaxis]

	## Subsampling to avoid number of parameters exploding
	X, X_test = X[:,::2], X_test[:,::2] 

	## Main variables
	learning_rate = 1e-4
	decay = learning_rate * 5
	train_split = 0.8
	## Splitting to get validation set
	X_train, X_valid, y_train, y_valid = split_data(X, y, train_split=train_split, shuffle=True)
	size = 10
	X_train, X_valid, y_train, y_valid = X_train[:size], X_valid[:size], y_train[:size], y_valid[:size]
	## Building and training model
	model = Sequential()
	n_epochs = 100
	early_stopping = EarlyStopping("valid_loss", 0.001, 10)
	model.add(Convo1D(3, 1, 32), "relu")
	model.add(MaxPool1D(2,2), "identity")
	model.add(Flatten(), "identity")
	model.add(layer=Linear(6272, 
						   100, 
						   init="xavier"), 
						   activation="relu")
	model.add(layer=Linear(100, 
						   10, 
						   init="xavier"), 
						   activation="softmax")
	model.compile(loss="sparse_categorical_crossentropy", 
				  optimizer="adam",
				  learning_rate=learning_rate,
				  metric="accuracy",
				  decay=decay)
	model.summary()
	model.fit(X_train, 
			  y_train, 
			  X_valid, 
			  y_valid, 
			  n_epochs=n_epochs, 
			  verbose=True,
			  early_stopping=early_stopping)
	model.plot_stats()
	## Show results
	if not latent:
		preds = np.argmax(model.predict(X_test), axis=1)
		for i in range(3):
			plt.imshow(X_test[i].reshape(width, height))
			plt.title(str("Prediction :" + label_name[preds[i]] + " Ground truth label :" + str(y_test[i])))
			plt.show()

loader = tf.keras.datasets.mnist
label_name = label_name_digits_mnist     
(X, y), (X_test, y_test) = loader.load_data()
execute_classification_model(X, y, X_test, y_test, label_name)