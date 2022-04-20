import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import nndiy.core
import nndiy.activation
import nndiy.loss
import nndiy.optimizer
import nndiy.layer


NN_METRIC_ARGS = (None, 'accuracy')
ACTI_MAPPING = dict(
	relu = nndiy.activation.ReLU,
	sigmoid = nndiy.activation.Sigmoid,
	tanh = nndiy.activation.Tanh,
	softmax = nndiy.activation.Softmax,
	identity = nndiy.activation.Identity,
	lrelu = nndiy.activation.LeakyReLU,
)
LOSS_MAPPING = dict(
	binary_crossentropy = nndiy.loss.BinaryCrossEntropy,
	categorical_crossentropy = nndiy.loss.CategoricalCrossEntropy,
	mse = nndiy.loss.MeanSquaredError,
	mae = nndiy.loss.MeanAbsoluteError,
	rmse = nndiy.loss.RootMeanSquaredError,
	sparse_binary_crossentropy = nndiy.loss.SparseBinaryCrossEntropy,
	sparse_categorical_crossentropy = nndiy.loss.SparseCategoricalCrossEntropy,
	sparse_categorical_crossentropy_softmax = nndiy.loss.SparseCategoricalCrossEntropySoftmax,
)
OPTI_MAPPING = dict(
	gd = nndiy.optimizer.GradientDescent,
	sgd = nndiy.optimizer.StochasticGradientDescent,
	mgd = nndiy.optimizer.MinibatchGradientDescent,
	adam = nndiy.optimizer.Adam,
)


class Sequential():
	"""The neural network's class consist of multiple linear/convolutional layers, their activation/pooling
	layers as well as a loss layer and an optimizer"""

	def __init__(self):
		self._net:List[nndiy.core.Module] = []

	def add(self, layer:nndiy.core.Module, activation=None):
		"""Add a layer to the network and its activation function"""
		assert isinstance(layer, nndiy.core.Module)
		self._net.append(layer)
		if activation in ACTI_MAPPING:
			self._net.append(ACTI_MAPPING[activation]())
		elif activation is not None:
			assert activation is None	# Raise an AssertionError when activation is invalid

	def compile(self, loss:str, optimizer:str, learning_rate=1e-3, metric=None, n_batch=20, decay=1e-6):
		"""Compile the network by adding the `loss` layer at the end using `optimizer`
		as learning method. Metric should be `accuracy` for classification problems otherwise
		left as `None`"""
		assert loss in LOSS_MAPPING
		assert optimizer in OPTI_MAPPING
		assert metric in NN_METRIC_ARGS
		assert n_batch > 0
		
		self._metric:str = metric
		self._net.append(LOSS_MAPPING[loss]())
		self._optim:nndiy.core.Optimizer = OPTI_MAPPING[optimizer](self, learning_rate, decay, n_batch)

	def fit(self, *args, n_epochs=20, verbose=True, early_stopping=None):
		"""Train the network using input data and truth values. It is also possible
		to pass validation data along with training data"""
		assert len(args) in (2, 4)
		if len(args) == 2:
			self._X, self._y = args
		else:
			self._X, self._y, self._valid_X, self._valid_y = args

		self._valid = (len(args) == 4)
		self._init_stats()
		self._optim.step(self._X, self._y, n_epochs, verbose, early_stopping)
	
	def predict(self, X:np.ndarray) -> np.ndarray:
		"""Make predictions on the dataset `X`"""
		return self._forward(X, None)

	def _forward(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
		"""Feed the data `X` through the network's layers to obtain the network's prediction"""
		for l in self._net[:-1]:
			l.forward(X)
			X = l._output	# since input is no longer needed, we re-use the variable X
		if y is not None:
			self._net[-1].forward(y, X)
		return X

	def _backward(self):
		"""Back propagate the output's gradient from the last to the first layer"""
		self._net[-1].backward()
		for i in range(len(self._net) - 1, 0, -1):
			self._net[i-1].backward(self._net[i]._grad_input)

	def _init_stats(self):
		self._train_loss_values = []
		self._valid_loss_values = []
		if self._metric == 'accuracy':
			self._train_acc_values = []
			self._valid_acc_values = []

	def _update_stats(self) -> Tuple[float, float, float, float]:
		train_loss, train_acc = self._compute_loss_accuracy(self._X, self._y)
		valid_loss, valid_acc = None, None
		self._train_loss_values.append(train_loss)
		if train_acc is not None:
			self._train_acc_values.append(train_acc)
		if self._valid:
			valid_loss, valid_acc = self._compute_loss_accuracy(self._valid_X, self._valid_y)
			self._valid_loss_values.append(valid_loss)
			if valid_acc is not None:
				self._valid_acc_values.append(valid_acc)
		return train_loss, train_acc, valid_loss, valid_acc

	def _compute_loss_accuracy(self, X:np.ndarray, y:np.ndarray) -> Tuple[float, float]:
		## Forward pass to update the network with X
		output = self._forward(X, y)
		loss = np.mean(self._net[-1]._output)
		acc = np.mean(y == np.argmax(output, axis=1)) if self._metric == 'accuracy' else None
		# loss = self._net[-1]._output.mean()
		# if self._metric == "accuracy":
		# 	acc = np.where(y == self._net[-2]._output.argmax(axis=1), 1, 0).mean()
		# else:
		# 	acc = None ## Trick to avoid displaying acc when not in classification
		return loss, acc

	def plot_stats(self):
		# Plot loss/epoch
		plt.plot(self._train_loss_values, label="train")
		if self._valid:
			plt.plot(self._valid_loss_values, label="valid")
		plt.legend(loc="upper right")
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.title("Loss per epoch")
		plt.show()

		# Plot accuracy/epoch
		if self._metric == 'accuracy':
			plt.plot(self._train_acc_values, label="train")
			if self._valid:
				plt.plot(self._valid_acc_values, label="valid")
			plt.legend(loc="lower right")
			plt.ylabel('Accuracy')
			plt.xlabel('Epochs')
			plt.title("Accuracy per epoch")
			plt.show()

	def summary(self):
		print("=" * 30)
		n = 1
		for l in self._net[:-1]:
			if isinstance(l, nndiy.layer.Linear):
				print(f"({n}) Linear layer with parameters of shape {l._W.shape}")
				n += 1
			elif isinstance(l, nndiy.layer.Convo1D):
				print(f"({n}) Convolution layer with parameters of shape {l._W.shape}")
				n += 1
			elif isinstance(l, nndiy.layer.MaxPool1D):
				print(f"  ---- Max Pooling layer")
			elif isinstance(l, nndiy.layer.Flatten):
				print(f"  ---- Flatten layer")
			elif isinstance(l, nndiy.layer.Dropout):
				print(f"({n}) Dropout layer with rate {l._rate}")
				n += 1
			elif isinstance(l, nndiy.core.Activation):
				print(f"  ---- Activation: {type(l)}")
		print(f"* Loss: {type(self._net[-1])}")
		print(f"* Optimizer: {type(self._optim)}")
		print(f"* Number of parameters: {self.count_params()}")
		print("=" * 30)

	def count_params(self) -> int:
		res = 0
		for l in self._net:
			if isinstance(l, nndiy.layer.Linear) or isinstance(l, nndiy.layer.Convo1D):
				res += l._W.size + l._b.size
		return res
