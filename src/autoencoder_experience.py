from global_imports import *
from utils.utils import *

label_name_fashion_mnist = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

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
	learning_rate = 1e-3
	decay = 1e-4
	regularization_lambda = 1e-6 ## L2 regularization
	n_epochs = 100
	train_split = 0.8
	early_stopping = {"patience": 5, "metric": "valid_loss", "min_delta": 0.001}
	## Splitting to get validation set
	X_train, X_valid, y_train, y_valid = split_data(X, y, train_split=train_split, shuffle=True)
	# size = 10_000
	# X_train, X_valid, y_train, y_valid = X_train[:size], X_valid[:size], y_train[:size], y_valid[:size]
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
				  optimizer="sgd",
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
	preds = model.predict(X_test)
	for i in range(3):
		plt.imshow(X_test[i].reshape(28,28))
		plt.title(str("Prediction :"+label_name_fashion_mnist[y_test[i]]))
		plt.show()

def mnist_reconstruction_mlp():
	## Loading fashion MNIST dataset
	fashion_mnist = tf.keras.datasets.fashion_mnist                            
	(X, _), (X_test, _) = fashion_mnist.load_data()
	## Reshaping to get linear data in order to use our feed forward model
	X = X.reshape(X.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)
	## Normalizing our data
	X = min_max_scaler(X, 0, 1)
	X_test = min_max_scaler(X_test, 0, 1)
	## Main variables
	n_features = X.shape[1]
	n_classes = X.shape[1]
	init_type = "xavier"
	learning_rate = 1e-3
	decay = 1e-3
	regularization_lambda = 1e-9 ## L2 regularization
	n_epochs = 100
	train_split = 0.8
	early_stopping = {"patience": 5, "metric": "valid_loss", "min_delta": 0.001}
	## Splitting to get validation set
	X_train, X_valid = split_X(X, train_split=train_split, shuffle=True)
	# size = 10_000
	# X_train, X_valid = X_train[:size], X_valid[:size]
	## Building and training model
	model = Sequential()
	model.add(layer=Linear(n_features, 
						   256, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="tanh")
	model.add(layer=Linear(256, 
						   64, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="tanh")
	model.add(layer=Linear(64, 
						   256, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="tanh")
	model.add(layer=Linear(256, 
						   n_classes, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="sigmoid")
	model.compile(loss="binary_crossentropy", 
				  optimizer="sgd",
				  learning_rate=learning_rate,
				  decay=decay)
	model.summary()
	model.fit(X_train, 
			  X_train, 
			  X_valid, 
			  X_valid, 
			  n_epochs=n_epochs, 
			  verbose=True,
			  early_stopping=early_stopping)
	model.plot_stats()
	preds = model.predict(X_test)
	for i in range(3):
		plt.imshow(X_test[i].reshape(28,28))
		plt.title("Input image")
		plt.show()
		plt.imshow(preds[i].reshape(28,28))
		plt.title("Reconstructed image")
		plt.show()

mnist_classification_mlp()
mnist_reconstruction_mlp()

# choice = "classification_mlp"
# choice = "reconstruction_autoencoder"

# if choice == "classification_mlp":
# 	mnist_classification_mlp()
# elif choice == "classification_cnn":
# 	pass
# elif choice == "reconstruction_autoencoder":
# 	mnist_reconstruction_mlp()
# elif choice == "classification_autoencoder":
# 	pass
	