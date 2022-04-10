from Core import *
from global_imports import *


class Sequential(Module):
	def __init__(self):
		self._network = []
		self._train_loss_values = []
		self._train_acc_values = []
		self._valid_loss_values = []
		self._valid_acc_values = []
		self._valid = False
		self._metric = None

	############################################### USER FUNCTIONS ###############################################

	def add(self, layer, activation=None):
		self._network.append(layer)
		if activation == "relu":
			self._network.append(ReLU())
		elif activation == "sigmoid":
			self._network.append(Sigmoid())
		elif activation == "tanh":
			self._network.append(Tanh())
		elif activation == "softmax":
			self._network.append(Softmax())
		elif activation == "linear":
			self._network.append(Lin())
		elif activation == "lrelu":
			self._network.append(LeakyReLU())

	def compile(self, loss, optimizer="sgd", learning_rate=1e-3, metric=None, n_batch=20, decay=1e-6):
		## Choosing a metric
		self._metric = metric
		## Choosing a loss function for our network
		if loss == "binary_crossentropy":
			loss_function = BinaryCrossEntropy()
		elif loss == "categorical_crossentropy":
			loss_function = CategoricalCrossEntropy()
		elif loss == "mse":
			loss_function = MeanSquaredError()
		elif loss == "mae":
			loss_function = MeanAbsoluteError()
		elif loss == "rmse":
			loss_function = RootMeanSquaredError()
		elif loss == "sparse_binary_crossentropy":
			loss_function = SparseBinaryCrossEntropy()
		elif loss == "sparse_categorical_crossentropy":
			output_activation = str(type(self._network[len(self._network) - 1]))
			if "Softmax" in output_activation:
				loss_function = SparseCategoricalCrossEntropySoftmax()
			else:
				loss_function = SparseCategoricalCrossEntropy()
		else:
			print("Error : wrong loss function")
		## Choosing and optimizer function for our network
		if optimizer == "gd":
			self.optimizer = GradientDescent(self, loss_function, learning_rate, decay=decay)
		elif optimizer == "sgd":
			self.optimizer = StochasticGradientDescent(self, loss_function, learning_rate, decay=decay)
		elif optimizer == "mgd":
			self.optimizer = MinibatchGradientDescent(self, loss_function, learning_rate, decay=decay, n_batch=n_batch)
		elif optimizer == "adam":
			self.optimizer = Adam(self, loss_function, learning_rate)
		else:
			print("Error : wrong optimizer")

	def fit(self, *arg, n_epochs=20, verbose=True, early_stopping=None):
		## If there is just a train set
		if len(arg) == 2:
			self._X = arg[0]
			self._y = arg[1]
		## If there is a train set and a valid set
		elif len(arg) == 4:
			self._X = arg[0]
			self._y = arg[1]
			self._valid_x = arg[2]
			self._valid_y = arg[3]
			self._valid = True
		else:
			print("Error : number of arguments in fit()")
		self.optimizer.step(self._X, self._y, n_epochs, verbose, early_stopping)
			
	def predict(self, X):
		last_module = len(self._network) - 1
		self._network[0].forward(X)
		for i in range(1, last_module):
			self._network[i].forward(self._network[i-1]._output)
		return self._network[last_module - 1]._output

	############################################### LOW LEVEL FUNCTIONS ###############################################

	def zero_grad(self):
		for m in self._network:
			if "layers" in str(type(m)):
				m.zero_grad()

	def forward(self, X, y):
		last_module = len(self._network) - 1
		## First forward pass on data <self._X>
		self._network[0].forward(X)
		## Forward on all the next layers
		for i in range(1, last_module):
			self._network[i].forward(self._network[i-1]._output)
		## Forward on the loss module which is the last module of the network
		self._network[last_module].forward(y, self._network[last_module - 1]._output)
		## Return the output of the last layer, before the loss module
		return self._network[last_module - 1]._output

	def update_parameters(self, learning_rate):
		for i in range(len(self._network) - 1):
			self._network[i].update_parameters(learning_rate)
			self._network[i].zero_grad()

	def backward(self):
		loss_function = self._network[len(self._network) - 1]
		## Backward pass on the loss function
		loss_function.backward()
		## Backward pass on every previous layers
		for i in range(len(self._network) - 1, 0, -1):
			self._network[i-1].backward_update_gradient(self._network[i]._new_delta)
			self._network[i-1].backward_delta()

	############################################### DISPLAY FUNCTIONS ###############################################

	def show_updates(self, cpt_epoch, learning_rate):
		print("epoch :", '{:>03g}'.format(cpt_epoch), end="")
		train_loss, train_acc = self.compute_train_score()
		if train_acc != 666 : ## Don't display when we are doing regression / autoencoder etc ...
			print(", train_acc :", '{:<08g}'.format(train_acc), end="")
		if train_loss < 0:
			print(", train_loss :", '{:<2e}'.format(train_loss), end="")
		else:
			print(", train_loss :", " " + ('{:<2e}'.format(train_loss)), end="")
		if self._valid: 
			valid_loss, valid_acc = self.compute_valid_score()
			if valid_acc != 666 : ## Don't display when we are doing regression / autoencoder etc ...
				print(", valid_acc :", '{:<08g}'.format(valid_acc), end="")
			if valid_loss < 0: 
				print(", valid_loss :", '{:<2e}'.format(valid_loss), end="")
			else:
				print(", valid_loss :", " " + ('{:<2e}'.format(valid_loss)), end="")
		print(", learning_rate :", '{:<2e}'.format(learning_rate), end="")
		print("")

	def update_stats(self):
		## Updates train tracking
		train_loss, train_acc = self.compute_train_score()
		self._train_loss_values.append(train_loss)
		self._train_acc_values.append(train_acc)
		## Updates valid tracking
		if self._valid:
			valid_loss, valid_acc = self.compute_valid_score()
			self._valid_loss_values.append(valid_loss)
			self._valid_acc_values.append(valid_acc)
			return train_loss, train_acc, valid_loss, valid_acc
		return train_loss, train_acc

	def compute_scores(self):
		train_loss, train_acc = self.compute_train_score(self, type)
		if self._valid:
			valid_loss, valid_acc = self.compute_valid_score(self, type)
			return train_loss, train_acc, valid_loss, valid_acc
		return train_loss, train_acc

	def compute_train_score(self):
		## Forward pass to update our network with train data
		self.forward(self._X, self._y)
		## Returns the loss and the metrics for the training data
		loss = self._network[len(self._network) - 1]._output.mean()
		if self._metric == "accuracy":
			acc = np.where(self._y == self._network[len(self._network) - 2]._output.argmax(axis=1), 1, 0).mean()
		else:
			acc = 666 ## Trick to avoid displaying acc when not in classification
		return loss, acc

	def compute_valid_score(self):
		## Forward pass to update our network with validation data
		## But we don't update parameters since its not our training data
		self.forward(self._valid_x, self._valid_y)
		## Returns the loss and the metrics for the validation data
		loss = self._network[len(self._network) - 1]._output.mean()
		if self._metric == "accuracy":
			acc = np.where(self._valid_y == self._network[len(self._network) - 2]._output.argmax(axis=1), 1, 0).mean()
		else:
			acc = 666 ## Trick to avoid displaying acc when not in classification
		return loss, acc

	def plot_stats(self):
		## Show loss per epochs
		lv_train = np.array(self._train_loss_values)
		lv_valid = np.array(self._valid_loss_values)
		plt.plot(lv_train, label="train")
		plt.plot(lv_valid, label="valid")
		plt.legend(loc="upper right")
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.title("Loss per epochs")
		plt.show()
		## Show accuracy per epochs
		la_train = np.array(self._train_acc_values)
		la_valid = np.array(self._valid_acc_values)
		plt.plot(la_train, label="train")
		plt.plot(la_valid, label="valid")
		plt.legend(loc="lower right")
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.title("Accuracy per epochs")
		plt.show()

	def summary(self):
		print("====================================================================================")
		print("==> Network :")
		for i, m in enumerate(self._network):
			type_ = str(type(m))
			element = type_.split(".")[-1][:-2]
			if "layers" in type_:
				if "Dropout" in type_:
					print("--> Layer", int(i/2), ":", element, "with rate =", m._rate)
					i += 1
				else:
					print("--> Layer", int(i/2), ":", element, "with parameters_shape =", m._parameters.shape, end="")
			elif "activation_functions" in type_:
				print(" and activation :", element)
			elif "loss_functions" in type_:
				print("* Loss :", element)
		print("* Optimizer :", str(type(self.optimizer)).split(".")[-1][:-2])
		print("* Total number of parameters :", self.count_parameters())
		print("====================================================================================")

	############################################### UTILITY FUNCTIONS ###############################################

	def count_parameters(self):
		res = 0
		for m in self._network:
			type_ = str(type(m))
			if "layers" in type_:
				if "Dropout" not in type_:
					res += m._parameters.size + m._bias.size
		return res