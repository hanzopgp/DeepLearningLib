from tkinter import Y
from global_imports import *
from utils.utils import *
import tensorflow as tf ## Usefull for datasets
np.random.seed(42)


label_name_fashion_mnist = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
label_name_digits_mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def execute_classification_model(X, y, X_test, y_test, label_name, latent=False):
	if not latent:
		width = X.shape[1]
		height = X.shape[2]
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
	regularization_lambda = 1e-9
	train_split = 0.8
	## Splitting to get validation set
	X_train, X_valid, y_train, y_valid = split_data(X, y, train_split=train_split, shuffle=True)
	# size = 10_000
	# X_train, X_valid, y_train, y_valid = X_train[:size], X_valid[:size], y_train[:size], y_valid[:size]
	## Building and training model
	model = Sequential()
	if not latent: ## Normal classification on 728 features MNIST
		n_epochs = 100
		early_stopping = {"patience": 5, "metric": "valid_loss", "min_delta": 0.001}
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
	## Show results
	if not latent:
		preds = np.argmax(model.predict(X_test), axis=1)
		for i in range(3):
			plt.imshow(X_test[i].reshape(width, height))
			plt.title(str("Prediction :" + label_name[preds[i]] + " Ground truth label :" + str(y_test[i])))
			plt.show()

def classification_mlp(dataset):
	## Loading dataset
	if dataset == "fashion_mnist":
		loader = tf.keras.datasets.fashion_mnist
		label_name = label_name_fashion_mnist 
	elif dataset == "digits_mnist":     
		loader = tf.keras.datasets.mnist
		label_name = label_name_digits_mnist     
	(X, y), (X_test, y_test) = loader.load_data()
	execute_classification_model(X, y, X_test, y_test, label_name)

def reconstruction_mlp(dataset, size_latent_space):
	## Loading fashion MNIST dataset
	if dataset == "fashion_mnist":
		loader = tf.keras.datasets.fashion_mnist
	elif dataset == "digits_mnist":     
		loader = tf.keras.datasets.mnist  
	(X, y), (X_test, y_test) = loader.load_data()
	width = X.shape[1]
	height = X.shape[2]
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
	learning_rate = 1e-4 ## 5e-4 works well for 64/16 latent space dim
	decay = 1e-4
	regularization_lambda = 1e-9 
	n_epochs = 100
	train_split = 0.8
	early_stopping = {"patience": 10, "metric": "valid_loss", "min_delta": 0.001}
	## Now we split to get validation set for the model training but this time we
	## don't need y labels for the autoencoder. Although it will be usefull to keep it
	## in order to train a classifier with the latent space representation made by our
	## autoencoder model.
	X_train, X_valid, _, _ = split_data(X, y, train_split=train_split, shuffle=True)
	size = 30_000
	X_train, X_valid, X_test, y_test = X_train[:size], X_valid[:size], X_test[:size], y_test[:size]
	## Building and training autoencoder model
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
						   size_latent_space, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="tanh")
	model.add(layer=Linear(size_latent_space, 
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
	## Show results
	preds = model.predict(X_test)
	for i in range(3):
		plt.imshow(X_test[i].reshape(width, height))
		plt.title("Input image")
		plt.show()
		plt.imshow(preds[i].reshape(width, height))
		plt.title("Reconstructed image")
		plt.show()
	## Once the model is trained and working we forward some data in the encoder layers
	## Thanks to that we get a compressed representation and we can see the performance
	## in classification. It should take way less time since we performed dimensionality
	## reduction. Here we have only two layers in the encoder so we will take some of our
	## test data and get their representation in our latent space.
	model._network[0].forward(X_test) ## This layer is the first linear one
	model._network[1].forward(model._network[0]._output) ## This layer is first activation function 
	model._network[2].forward(model._network[1]._output) ## This layer is the second linear layer
	model._network[3].forward(model._network[2]._output) ## This layer is the second activation function
	model._network[4].forward(model._network[3]._output) ## This layer is the third linear layer
	model._network[5].forward(model._network[4]._output) ## This layer is the third activation function
	latent_space_X = model._network[5]._output ## Here we get the output and it's already normalized by tanh
	latent_space_y = y_test ## We kept track of the labels in order to train the next classifier
	print("X shape before compression :", X_test.shape)
	print("X shape after compression :", latent_space_X.shape)
	## After that we split our latent space data
	X_train, X_test, y_train, y_test = split_data(latent_space_X, latent_space_y, train_split=train_split, shuffle=True)
	## And we train a classifier with the compressed data
	execute_classification_model(X_train, y_train, X_test, y_test, label_name=label_name_digits_mnist, latent=True)

def build_noisy_mnist_digits_dataset(X, X_test, noise_amount, show=False):
	X_noise = X.copy()
	X_test_noise = X_test.copy()
	if show:
		for img in X[:5]:
			plt.imshow(img.reshape(28, 28))
			plt.show()
	print("--> Adding noise to X_train")
	for i in tqdm(range(X.shape[0])):
		for j in range(X.shape[1]):
			if np.random.rand(1) > 1-noise_amount:
				X_noise[i][j] = np.random.rand()*255
	print("--> Adding noise to X_test")
	for i in tqdm(range(X_test.shape[0])):
		for j in range(X_test.shape[1]):
			if np.random.rand(1) > 1-noise_amount:
				X_test_noise[i][j] = np.random.rand()*255
	if show:			
		for img in X[:5]:
			plt.imshow(img.reshape(28, 28))
			plt.show()
	return X_noise, X_test_noise

def remove_noise_autoencoder(dataset, noise_amount):
	## Loading fashion MNIST dataset
	if dataset == "fashion_mnist":
		loader = tf.keras.datasets.fashion_mnist
	elif dataset == "digits_mnist":     
		loader = tf.keras.datasets.mnist  
	(X, _), (X_test, _) = loader.load_data()
	# size = 30_000
	# X, X_test = X[:size], X_test[:size]
	width = X.shape[1]
	height = X.shape[2]
	## Reshaping to get linear data in order to use our feed forward model
	X = X.reshape(X.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)
	## Adding noise to dataset
	X_noise, X_test_noise = build_noisy_mnist_digits_dataset(X, X_test, noise_amount)
	## Normalizing our data
	X = min_max_scaler(X, 0, 1)
	X_test = min_max_scaler(X_test, 0, 1)
	X_noise = min_max_scaler(X_noise, 0, 1)
	X_test_noise = min_max_scaler(X_test_noise, 0, 1)
	## Main variables
	n_features = X.shape[1]
	n_classes = X.shape[1]
	init_type = "xavier"
	learning_rate = 1e-4 ## Test 1e-3 comme pr 0.1
	decay = learning_rate*10
	regularization_lambda = 1e-9 
	n_epochs = 100
	train_split = 0.8
	early_stopping = {"patience": 10, "metric": "valid_loss", "min_delta": 0.001}
	X_train_noise, X_valid_noise = split_X(X_noise, train_split=train_split, shuffle=False)
	X_train, X_valid = split_X(X, train_split=train_split, shuffle=False)
	## Building and training autoencoder model
	model = Sequential()
	model.add(layer=Linear(n_features, 
						   256, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="relu")
	model.add(layer=Linear(256, 
						   180, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="relu")
	model.add(layer=Linear(180, 
						   128, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="relu")
	model.add(layer=Linear(128, 
						   180, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="relu")
	model.add(layer=Linear(180, 
						   256, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="relu")
	model.add(layer=Linear(256, 
						   n_classes, 
						   init_type=init_type, 
						   regularization_lambda=regularization_lambda), activation="sigmoid")
	model.compile(loss="binary_crossentropy", 
				  optimizer="sgd",
				  learning_rate=learning_rate,
				  decay=decay)
	model.summary()
	model.fit(X_train_noise, 
			  X_train, 
			  X_valid_noise, 
			  X_valid, 
			  n_epochs=n_epochs, 
			  verbose=True,
			  early_stopping=early_stopping)
	model.plot_stats()
	preds = model.predict(X_test_noise)
	## Show results
	for i in range(3):
		plt.imshow(X_test_noise[i].reshape(width, height))
		plt.title("Input image")
		plt.show()
		plt.imshow(preds[i].reshape(width, height))
		plt.title("Reconstructed image")
		plt.show()
		plt.imshow(X_test[i].reshape(width, height))
		plt.title("Ground truth image")
		plt.show()
	

## Classic classification with a MLP model
# classification_mlp("fashion_mnist")
# classification_mlp("digits_mnist")

## Autoencoder reconstruction + classification with latent space
# reconstruction_mlp("fashion_mnist", 16)
# reconstruction_mlp("digits_mnist", 8) ## RETRY WITH 8 need train.png and network.png

## Autoencoder to remove noise 
# remove_noise_autoencoder("fashion_mnist", 0.1)
remove_noise_autoencoder("digits_mnist", 0.7)