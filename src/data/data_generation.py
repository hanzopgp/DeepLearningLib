import numpy as np
from matplotlib import pyplot as plt

class DataGeneration:

	def __init__(self, points, classes):
		self.points = points
		self.classes = classes
		self.x = []
		self.y = [] 

	def make_vertical_data(self):
		x = np.zeros((self.points*self.classes, 2))
		y = np.zeros(self.points*self.classes, dtype='uint8')
		for class_number in range(self.classes):
			ix = range(self.points*class_number, self.points*(class_number+1))
			x[ix] = np.c_[np.random.randn(self.points)*.1 + (class_number)/3, np.random.randn(self.points)*.1 + 0.5]
			y[ix] = class_number
		self.x = x
		self.y = y

	def make_spiral_data(self):
		x = np.zeros((self.points*self.classes, 2))
		y = np.zeros(self.points*self.classes, dtype='uint8')
		for class_number in range(self.classes):
			ix = range(self.points*class_number, self.points*(class_number+1))
			r = np.linspace(0.0, 1, self.points)
			t = np.linspace(class_number*4, (class_number+1)*4, self.points) + (np.random.randn(self.points) * 0.2)
			x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
			y[ix] = class_number
		self.x = x
		self.y = y

	def make_sinus_data(self):
		eps = (np.random.randn(self.points).reshape(-1, 1))*.1 + 0.5
		x = np.arange(self.points).reshape(-1, 1) / self.points - eps
		y = np.sin(2 * np.pi * x).reshape(-1, 1) - eps
		self.x = x
		self.y = y

	def display_data(self):
		if self.x.shape[1] > 1:
			plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap="brg")
			plt.show()
		else:
			plt.scatter(self.x, self.y)
			plt.show()