import numpy as np
import matplotlib.pyplot as plt

class DataGeneration:

	def __init__(self, points, classes):
		self.points = points
		self.classes = classes
		self.x = []
		self.y = [] 

	def make_vertical_data(self):
		x = np.zeros((samples*self.classes, 2))
		y = np.zeros(samples*self.classes, dtype='uint8')
		for class_number in range(self.classes):
			ix = range(samples*class_number, samples*(class_number+1))
			x[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
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

	def make_sinus_data(self, samples=1000):
		self.x = np.arange(samples).reshape(-1, 1) / samples
		self.y = np.sin(2 * np.pi * X).reshape(-1, 1)

	def display_data(self):
		plt.scatter(self.x[:, 0], self.x[:, 1])
		plt.show()
		plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap="brg")
		plt.show()