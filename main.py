import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nndiy import Sequential
from nndiy.layer import Linear
from nndiy.convolution import Convo1D, MaxPool1D, Flatten
from nndiy.early_stopping import EarlyStopping
from nndiy.utils import min_max_scaler
# from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

np.random.seed(42)

label_name_digits_mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def execute_classification_model(X, y, X_test, y_test, label_name, latent=False):
	if not latent:
		width = X.shape[1]
		height = X.shape[2]
	## Reshaping to get linear data in order to use our feed forward model
	# X = X.reshape(*(X.shape), 1)
	# X_test = X_test.reshape(*(X.shape), 1)
	## Normalizing our data
	X = min_max_scaler(X, 0, 1)
	X_test = min_max_scaler(X_test, 0, 1)

	X = X.reshape(X.shape[0], -1)[:,:, np.newaxis]
	X_test = X.reshape(X_test.shape[0], -1)[:,:, np.newaxis]

	## Main variables
	n_features = X.shape[1]
	n_classes = len(np.unique(y))	
	init_type = "xavier"
	learning_rate = 1e-4
	decay = learning_rate * 5
	regularization_lambda = 1e-9
	train_split = 0.8
	## Splitting to get validation set
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_split, shuffle=True)
	size = 1_000
	X_train, X_valid, y_train, y_valid = X_train[:size], X_valid[:size], y_train[:size], y_valid[:size]
	## Building and training model


	model = Sequential()
	if not latent: ## Normal classification on 728 features MNIST
		n_epochs = 100
		# early_stopping = {"patience": 10, "metric": "valid_loss", "min_delta": 0.001}
		early_stopping = EarlyStopping("valid_loss", 0.001, 10)
		model.add(Convo1D(3, 1, 32), "relu")
		model.add(MaxPool1D(2,2), "identity")
		model.add(Flatten(), "identity")
		model.add(layer=Linear(12512, 
							100, 
							init="xavier"), activation="relu")
		model.add(layer=Linear(100, 
							10, 
							init="xavier"), activation="softmax")
		# model.add(layer=Linear(256, 
		# 					128, 
		# 					init_type=init_type, 
		# 					regularization_lambda=regularization_lambda), activation="tanh")
		# model.add(layer=Linear(128, 
		# 					n_classes, 
		# 					init_type=init_type, 
		# 					regularization_lambda=regularization_lambda), activation="softmax")
	else: ## Latent classification on 64 features MNIST
		n_epochs = 200
		early_stopping = {"patience": 15, "metric": "valid_loss", "min_delta": 0.001}
		model.add(layer=Linear(n_features, 
							32, 
							init_type=init_type, 
							regularization_lambda=regularization_lambda), activation="tanh")
		model.add(layer=Linear(32, 
							16, 
							init_type=init_type, 
							regularization_lambda=regularization_lambda), activation="tanh")
		model.add(layer=Linear(16, 
							n_classes, 
							init_type=init_type, 
							regularization_lambda=regularization_lambda), activation="softmax")	
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