from Core import *
from global_imports import *


class Sequential(Module):
	def __init__(self):
		self.network = []
		self._train_loss_values = []
		self._train_acc_values = []
		self._valid_loss_values = []
		self._valid_acc_values = []
		self._valid = False
		self._metric = None

	############################################### USER FUNCTIONS ###############################################

	def add(self, layer, activation):
		self.network.append(layer)
		if activation == "relu":
			self.network.append(ReLU())
		elif activation == "sigmoid":
			self.network.append(Sigmoid())
		elif activation == "tanh":
			self.network.append(Tanh())
		elif activation == "softmax":
			self.network.append(Softmax())
		else:
			print("Error : wrong activation function")

	def compile(self, loss, optimizer, learning_rate, metric, n_batch=0, decay=1e-6):
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
			output_activation = str(type(self.network[len(self.network) - 1]))
			if "Softmax" in output_activation:
				loss_function = SparseCategoricalCrossEntropySoftmax()
			else:
				loss_function = SparseCategoricalCrossEntropy()
		else:
			print("Error : wrong loss function")
		## Choosing and optimizer function for our network
		if optimizer == "GD":
			self.optimizer = GradientDescent(self, loss_function, learning_rate)
		elif optimizer == "SGD":
			self.optimizer = StochasticGradientDescent(self, loss_function, learning_rate, decay=decay)
		elif optimizer == "MGD":
			self.optimizer = MinibatchGradientDescent(self, loss_function, learning_rate, n_batch=n_batch)
		else:
			print("Error : wrong optimizer")

	def fit(self, *arg, n_epochs, verbose):
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
		self.optimizer.step(self._X, self._y, n_epochs, verbose)
			
	def predict(self, X):
		last_module = len(self.network) - 1
		self.network[0].forward(X)
		for i in range(1, last_module):
			self.network[i].forward(self.network[i-1]._output)
		return self.network[last_module - 1]._output

	############################################### LOW LEVEL FUNCTIONS ###############################################

	def zero_grad(self):
		for m in self.network:
			if "layers" in str(type(m)):
				m.zero_grad()

	def forward(self, X, y):
		last_module = len(self.network) - 1
		## First forward pass on data <self._X>
		self.network[0].forward(X)
		## Forward on all the next layers
		for i in range(1, last_module):
			self.network[i].forward(self.network[i-1]._output)
		## Forward on the loss module which is the last module of the network
		self.network[last_module].forward(y, self.network[last_module - 1]._output)
		## Return the output of the last layer, before the loss module
		return self.network[last_module - 1]._output

	def update_parameters(self, learning_rate):
		for i in range(len(self.network) - 1):
			self.network[i].update_parameters(learning_rate)

	def backward(self):
		loss_function = self.network[len(self.network) - 1]
		## Backward pass on the loss function
		loss_function.backward()
		## Backward pass on every previous layers
		for i in range(len(self.network) - 1, 0, -1):
			self.network[i-1].backward_update_gradient(self.network[i]._new_delta)
			self.network[i-1].backward_delta()

	############################################### DISPLAY FUNCTIONS ###############################################

	def show_updates(self, cpt_epoch, learning_rate):
		print("epoch :", '{:>03g}'.format(cpt_epoch), end="")
		train_loss, train_acc = self.compute_train_score()
		print(", train_acc :", '{:<08g}'.format(train_acc), end="")
		if train_loss < 0:
			print(", train_loss :", '{:<2e}'.format(train_loss), end="")
		else:
			print(", train_loss :", " "+('{:<2e}'.format(train_loss)), end="")
		if self._valid: 
			valid_loss, valid_acc = self.compute_valid_score()
			print(", valid_acc :", '{:<08g}'.format(valid_acc), end="")
			if valid_loss < 0: 
				print(", valid_loss :", '{:<2e}'.format(valid_loss), end="")
			else:
				print(", valid_loss :", " "+('{:<2e}'.format(valid_loss)), end="")
		print(", learning_rate :", '{:<2e}'.format(learning_rate), end="")
		print("")

	def update_stats(self):
		## Updates train tracking
		loss, acc = self.compute_train_score()
		self._train_loss_values.append(loss)
		self._train_acc_values.append(acc)
		## Updates valid tracking
		if self._valid:
			loss, acc = self.compute_valid_score()
			self._valid_loss_values.append(loss)
			self._valid_acc_values.append(acc)

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
		loss = self.network[len(self.network) - 1]._output.mean()
		if self._metric == "accuracy":
			acc = np.where(self._y == self.network[len(self.network) - 2]._output.argmax(axis=1), 1, 0).mean()
		return loss, acc

	def compute_valid_score(self):
		## Forward pass to update our network with validation data
		## But we don't update parameters since its not our training data
		self.forward(self._valid_x, self._valid_y)
		## Returns the loss and the metrics for the validation data
		loss = self.network[len(self.network) - 1]._output.mean()
		if self._metric == "accuracy":
			acc = np.where(self._valid_y == self.network[len(self.network) - 2]._output.argmax(axis=1), 1, 0).mean()
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
		for i, m in enumerate(self.network):
			type_ = str(type(m))
			element = type_.split(".")[-1][:-2]
			if "layers" in type_:
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
		for m in self.network:
			if "layers" in str(type(m)):
				res += m._parameters.size + m._bias.size
		return res