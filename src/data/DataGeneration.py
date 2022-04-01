import numpy as np
from matplotlib import pyplot as plt

class DataGeneration:
	def __init__(self, nb_points):
		self.x, self.y = None, None
		self.nb_points = nb_points

	def display_data(self):
		if self.x is None or self.y is None:
			raise ValueError("Data is not generated. Nothing to display")
		print("X", self.x.shape, "Y", self.y.shape)
		if self.x.shape[1] > 1:
			plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap="brg")
			plt.show()
		else:
			plt.scatter(self.x, self.y)
			plt.show()


class DataGenMultiClass(DataGeneration):
	def __init__(self, nb_classes, nb_points=1000):
		super().__init__(nb_points=nb_points)
		self.nb_classes = nb_classes

	def make_vertical(self):
		class_size = self.nb_points // self.nb_classes
		self.x = np.zeros((class_size * self.nb_classes, 2))
		self.y = np.zeros(class_size * self.nb_classes, dtype=np.uint8)
		for cl in range(self.nb_classes):
			ix = range(class_size * cl, class_size * (cl+1))
			self.x[ix] = np.c_[np.random.randn(class_size)/10 + cl/3, np.random.randn(class_size)/10 + 0.5]
			self.y[ix] = cl

	def make_spiral(self):
		class_size = self.nb_points // self.nb_classes
		self.x = np.zeros((class_size * self.nb_classes, 2))
		self.y = np.zeros(class_size * self.nb_classes, dtype=np.uint8)
		for cl in range(self.nb_classes):
			ix = range(class_size * cl, class_size * (cl+1))
			r = np.linspace(0, 1, class_size)
			t = np.linspace(cl * 4, (cl+1) * 4, class_size) + np.random.randn(class_size)*0.2
			self.x[ix] = np.c_[r * np.sin(t*2.5), r * np.cos(t*2.5)]
			self.y[ix] = cl

	def make_sinus(self):
		eps = np.random.randn(self.nb_points).reshape(-1, 1) / 10 + 0.5
		self.x = np.arange(self.nb_points).reshape(-1, 1) / self.nb_points - eps
		self.y = np.sin(2 * np.pi * self.x).reshape(-1, 1) - eps


class DataGen2Classes(DataGenMultiClass):
	def __init__(self, nb_points=1000, eps=0.02):
		super().__init__(nb_classes=2, nb_points=nb_points)
		self.eps = eps
	
	def make_2_gaussians(self, center_x=1, sigma=0.1):
		x_pos = np.random.multivariate_normal(
			[center_x, center_x],
			np.diag([sigma, sigma]),
			self.nb_points // 2)
		x_neg = np.random.multivariate_normal(
			[-center_x, -center_x],
			np.diag([sigma, sigma]),
			self.nb_points // 2)
		self.x = np.vstack((x_pos, x_neg))
		self.y = np.hstack(
			(np.ones(self.nb_points // 2, dtype=np.uint8),
			-np.ones(self.nb_points // 2, dtype=np.uint8)))
		self._mix_data()

	def make_4_gaussians(self, center_x=1, sigma=0.1):
		xpos = np.vstack(
			(np.random.multivariate_normal(
				[center_x, center_x],
				np.diag([sigma,sigma]),
				self.nb_points // 4),
			np.random.multivariate_normal(
				[-center_x, -center_x],
				np.diag([sigma, sigma]),
				self.nb_points // 4)))
		xneg = np.vstack(
			(np.random.multivariate_normal(
				[-center_x, center_x],
				np.diag([sigma, sigma]),
				self.nb_points // 4),
			np.random.multivariate_normal(
				[center_x, -center_x],
				np.diag([sigma, sigma]),
				self.nb_points // 4)))
		self.x = np.vstack((xpos,xneg))
		self.y = np.hstack(
			(np.ones(self.nb_points // 2, dtype=np.uint8),
			-np.ones(self.nb_points // 2, dtype=np.uint8)))
		self._mix_data()

	def make_checker_board(self):
		self.x = np.random.uniform(-4 , 4, 2*self.nb_points).reshape((self.nb_points, 2))
		y = np.ceil(self.x[:,0]) + np.ceil(self.x[:,1])
		self.y = np.array(2*(y % 2) - 1, dtype=np.uint8)
		self._mix_data()

	def _mix_data(self):
		self.x[:,0] += np.random.normal(0, self.eps, self.nb_points)
		self.x[:,1] += np.random.normal(0, self.eps, self.nb_points)
		# Shuffle the data
		idx = np.random.permutation((range(self.y.size)))
		self.x = self.x[idx, :]
		self.y = self.y[idx].reshape(-1, 1)

