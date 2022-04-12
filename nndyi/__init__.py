import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import nndyi.core
import nndyi.activation
import nndyi.loss
import nndyi.optimizer
import nndyi.layer


NN_METRIC_ARGS = (None, 'accuracy')
ACTI_MAPPING = dict(
	relu = nndyi.activation.ReLU,
	sigmoid = nndyi.activation.Sigmoid,
	tanh = nndyi.activation.Tanh,
	softmax = nndyi.activation.Softmax,
	identity = nndyi.activation.Identity,
	lrelu = nndyi.activation.LeakyReLU,
)
LOSS_MAPPING = dict(
	binary_crossentropy = nndyi.loss.BinaryCrossEntropy,
	categorical_crossentropy = nndyi.loss.CategoricalCrossEntropy,
	mse = nndyi.loss.MeanSquaredError,
	mae = nndyi.loss.MeanAbsoluteError,
	rmse = nndyi.loss.RootMeanSquaredError,
	sparse_binary_crossentropy = nndyi.loss.SparseBinaryCrossEntropy,
	sparse_categorical_crossentropy = nndyi.loss.SparseCategoricalCrossEntropy,
	sparse_categorical_crossentropy_softmax = nndyi.loss.SparseCategoricalCrossEntropySoftmax,
)
OPTI_MAPPING = dict(
	gd = nndyi.optimizer.GradientDescent,
	sgd = nndyi.optimizer.StochasticGradientDescent,
	mgd = nndyi.optimizer.MinibatchGradientDescent,
	adam = nndyi.optimizer.Adam,
)


class Sequential():
	"""The neural network's class consist of multiple linear/convolutional layers, their activation/pooling
	layers as well as a loss layer and an optimizer"""

	def __init__(self):
		self._net:List[nndyi.core.Module] = []

	def add(self, layer:nndyi.core.Module, activation=None):
		"""Add a layer to the network and its activation function"""
		assert isinstance(layer, nndyi.core.Module)
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
		self._optim:nndyi.core.Optimizer = OPTI_MAPPING[optimizer](self, learning_rate, decay, n_batch)

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
		for i in range(len(self._net) - 1, 0, -1):
			self._net[i].backward()
			delta = self._net[i]._new_delta
			self._net[i-1].backward_update_gradient(delta)

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
		acc = np.mean(self._y == np.argmax(output, axis=1)) if self._metric == 'accuracy' else None
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
			if isinstance(l, nndyi.layer.Linear):
				print(f"({n}) Linear layer with parameters of shape {l._W.shape}")
				n += 1
			elif isinstance(l, nndyi.layer.Dropout):
				print(f"({n}) Dropout layer with rate {l._rate}")
				n += 1
			elif isinstance(l, nndyi.core.Activation):
				print(f"  ---- Activation: {type(l)}")
		print(f"* Loss: {type(self._net[-1])}")
		print(f"* Optimizer: {type(self._optim)}")
		print(f"* Number of parameters: {self.count_params()}")
		print("=" * 30)

	def count_params(self) -> int:
		res = 0
		for l in self._net[-1]:
			if isinstance(l, nndyi.layer.Linear):
				res += l._W.size + l._b.size
		return res