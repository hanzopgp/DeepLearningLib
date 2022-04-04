import numpy as np
from matplotlib import pyplot as plt

class DataGeneration:
	def __init__(self, nb_points, eps):
		self.x, self.y = None, None
		self.nb_points = nb_points
		self.eps = eps

	def display_data(self):
		if self.x is None or self.y is None:
			raise ValueError("Data is not generated. Nothing to display")
		if self.x.shape[1] > 1:
			plt.scatter(self.x[:, 0], self.x[:, 1], marker='.', c=self.y, cmap="brg")
			plt.show()
		else:
			plt.scatter(self.x, self.y, marker='.')
			plt.show()


class ContinuousGen(DataGeneration):
	def __init__(self, nb_points=1000, eps=0.1, sigma=0.1):
		super().__init__(nb_points=nb_points, eps=eps)
		self.sigma = sigma

	def make_sinus(self, freq=2, ampli=1, affine=0.9):
		self.x = np.linspace(0, np.pi, self.nb_points).reshape(-1, 1) * freq
		self.y = np.sin(self.x + affine) * ampli
		self._mix_data()

	def make_regression(self, slope=1, affine=0):
		self.x = np.linspace(-2, 2, self.nb_points).reshape(-1, 1)
		self.y = self.x * slope + affine
		self._mix_data()

	def _mix_data(self):
		self.y += np.random.normal(0, self.sigma, self.x.shape)
		idx = np.random.permutation((range(self.y.size)))
		self.x = self.x[idx,:]
		self.y = self.y[idx,:]


class MultiClassGen(DataGeneration):
	def __init__(self, nb_classes, nb_points=1000, eps=0.1):
		super().__init__(nb_points=nb_points, eps=eps)
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


class TwoClassGen(MultiClassGen):
	def __init__(self, nb_points=1000, eps=0.1):
		super().__init__(nb_classes=2, nb_points=nb_points, eps=eps)
	
	def make_2_gaussians(self, center_x=1, sigma=0.1):
		x_one = np.random.multivariate_normal(
			[center_x, center_x],
			np.diag([sigma, sigma]),
			self.nb_points // 2)
		x_zero = np.random.multivariate_normal(
			[-center_x, -center_x],
			np.diag([sigma, sigma]),
			self.nb_points // 2)
		self.x = np.vstack((x_one, x_zero))
		self.y = np.hstack(
			(np.ones(self.nb_points // 2, dtype=np.uint8),
			np.zeros(self.nb_points // 2, dtype=np.uint8)))
		self._mix_data()

	def make_4_gaussians(self, center_x=1, sigma=0.1):
		x_one = np.vstack(
			(np.random.multivariate_normal(
				[center_x, center_x],
				np.diag([sigma,sigma]),
				self.nb_points // 4),
			np.random.multivariate_normal(
				[-center_x, -center_x],
				np.diag([sigma, sigma]),
				self.nb_points // 4)))
		x_zero = np.vstack(
			(np.random.multivariate_normal(
				[-center_x, center_x],
				np.diag([sigma, sigma]),
				self.nb_points // 4),
			np.random.multivariate_normal(
				[center_x, -center_x],
				np.diag([sigma, sigma]),
				self.nb_points // 4)))
		self.x = np.vstack((x_one,x_zero))
		self.y = np.hstack(
			(np.ones(self.nb_points // 2, dtype=np.uint8),
			np.zeros(self.nb_points // 2, dtype=np.uint8)))
		self._mix_data()

	def make_checker_board(self):
		self.x = np.random.uniform(-4 , 4, 2*self.nb_points).reshape((self.nb_points, 2))
		y = np.ceil(self.x[:,0]) + np.ceil(self.x[:,1])
		self.y = np.array(y % 2, dtype=np.uint8)
		self._mix_data()

	def _mix_data(self):
		self.x[:,0] += np.random.normal(0, self.eps, self.nb_points)
		self.x[:,1] += np.random.normal(0, self.eps, self.nb_points)
		idx = np.random.permutation((range(self.y.size)))
		self.x = self.x[idx, :]
		self.y = self.y[idx].reshape(-1, 1)
