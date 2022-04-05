from global_imports import *
from utils.utils import *


def mnist_classification_mlp():
	## Loading fashion MNIST dataset
	fashion_mnist = tf.keras.datasets.fashion_mnist                            
	(X, y), (X_test, y_test) = fashion_mnist.load_data()
	## Reshaping to get linear data in order to use our feed forward model
	X = X.reshape(X.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)
	## Normalizing our data
	X = min_max_scaler(X, 0, 1)
	X_test = min_max_scaler(X_test, 0, 1)
	## Main variables
	n_features = X.shape[1]
	n_classes = len(np.unique(y))	
	init_type = "xavier"
	learning_rate = 1e-2
	decay = 1e-2
	regularization_lambda = 1e-6 ## L2 regularization
	n_epochs = 30
	train_split = 0.8
	## Splitting to get validation set
	X_train, X_valid, y_train, y_valid = split_data(X, y, train_split=train_split, shuffle=True)
	size = 5_000
	X_train, X_valid, y_train, y_valid = X_train[:size], X_valid[:size], y_train[:size], y_valid[:size]

	# model = tf.keras.models.Sequential()
	# model.add(tf.keras.layers.Dense(256, activation="tanh")) 
	# model.add(tf.keras.layers.Dense(128, activation="tanh")) 
	# model.add(tf.keras.layers.Dense(10, activation="softmax")) 
	# model.compile(
	# 	loss="sparse_categorical_crossentropy",
	# 	optimizer="sgd",                        
	# 	metrics=["accuracy"]                   
	# )
	# history = model.fit(X_train, y_train, epochs=10, validation_split=0.2) 

	## Building and training model
	model = Sequential()
	model.add(layer=Linear(n_features, 
						   256, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="tanh")
	model.add(layer=Linear(256, 
						   128, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="tanh")
	model.add(layer=Linear(128, 
						   n_classes, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="softmax")
	model.compile(loss="sparse_categorical_crossentropy", 
				  optimizer="SGD",
				  learning_rate=learning_rate,
				  metric="accuracy",
				  decay=decay)
	model.summary()
	model.fit(X_train, 
			  y_train, 
			  X_valid, 
			  y_valid, 
			  n_epochs=n_epochs, 
			  verbose=True)
	model.plot_stats()


choice = "classification_mlp"

if choice == "classification_mlp":
	mnist_classification_mlp()
elif choice == "classification_cnn":
	pass
elif choice == "reconstruction_autoencoder":
	pass
elif choice == "classification_autoencoder":
	pass
	