from collections import namedtuple
from core import Optimizer


EarlyStopping = namedtuple("EarlyStopping", ('metric', 'min_delta', 'patience'))


class GradientDescent(Optimizer):
	pass


class StochasticGradientDescent(Optimizer):
	pass


class MinibatchGradientDescent(Optimizer):
	pass


class Adam(Optimizer):
	pass

