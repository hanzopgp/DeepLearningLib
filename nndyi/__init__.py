from activation import *
from loss import *
from optimizer import *
from typing import List


ACTI_MAPPING = dict(
	relu = ReLU,
	sigmoid = Sigmoid,
	tanh = Tanh,
	softmax = Softmax,
	identity = Identity,
	lrelu = LeakyReLU,
)
LOSS_MAPPING = dict(
	binary_crossentropy = BinaryCrossEntropy,
	categorical_crossentropy = CategoricalCrossEntropy,
	mse = MeanSquaredError,
	mae = MeanAbsoluteError,
	rmse = RootMeanSquaredError,
	sparse_binary_crossentropy = SparseBinaryCrossEntropy,
	sparse_categorical_crossentropy = SparseCategoricalCrossEntropy,
	sparse_categorical_crossentropy_softmax = SparseCategoricalCrossEntropySoftmax,
)
OPTI_MAPPING = dict(
	gd = GradientDescent,
	sgd = StochasticGradientDescent,
	mgd = MinibatchGradientDescent,
	adam = Adam,
)


class Linear(Module):
	pass


class Sequential():
	"""The neural network's class consist of multiple linear/convolutional layers, their activation/pooling
	layers as well as a loss layer and an optimizer"""
	def __init__(self, layers=[]):
		# First check if layers list contains only valid network modules
		for layer in layers:
			assert isinstance(layer, Module)
		self._layers:List[Module] = layers

	def add(self, layer:Module, activation=None):
		"""Add a layer to the network and its activation function"""
		assert isinstance(layer, Module)
		self._layers.append(layer)
		if activation in ACTI_MAPPING:
			self._layers.append(ACTI_MAPPING[activation]())
		elif activation is not None:
			assert activation is None	# Raise an AssertionError when activation is invalid

	def compile(self, loss:str, optimizer:str, learning_rate=1e-3, metric=None, n_batch=20, decay=1e-6):
		"""Compile the network by adding the `loss` layer at the end using `optimizer`
		as learning method. Metric should be `accuracy` for classification problems otherwise
		left as `None`"""
		assert loss in LOSS_MAPPING
		assert optimizer in OPTI_MAPPING
		assert metric in (None, 'accuracy')
		assert n_batch <= 0
		
		self._metric:str = metric
		self._layers.append(LOSS_MAPPING[loss]())
		self._optim:Optimizer = OPTI_MAPPING[optimizer](learning_rate, decay, n_batch)

	def fit(self, *args, n_epochs=20, verbose=True, early_stopping:EarlyStopping=None):
		"""Train the network using input data and truth values. It is also possible
		to pass validation data along with training data"""
		assert len(args) in (2, 4)
		if len(args) == 2:
			self._X, self._y = args
		else:
			self._X, self._y, self._valid_X, self._valid_y = args
		self._valid = (len(args) == 4)
		# self._optim.step(self._X, self._y, n_epochs, verbose, early_stopping)

	def zero_grad(self):
		"""Resets gradient for every layer in the network"""
		for layer in self._layers:
			layer.zero_grad()

	
