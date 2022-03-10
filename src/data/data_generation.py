import numpy as np
import matplotlib.pyplot as plt

class DataExample:

    def __init__(self, points, classes):
        self.points = points
        self.classes = classes
        self.x = []
        self.y = [] 

    def make_spiral_data(self):
        x = np.zeros((self.points*self.classes, 2))
        y = np.zeros(self.points*self.classes, dtype='uint8')
        for class_nb in range(self.classes):
            ix = range(self.points*class_nb, self.points*(class_nb+1))
            r = np.linspace(0.0, 1, self.points)
            t = np.linspace(class_nb*4, (class_nb+1)*4, self.points) + (np.random.randn(self.points) * 0.2)
            x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = class_nb
        self.x = x
        self.y = y

    def make_vertical_data(self):
        x, y = vertical_data(self.points, self.classes)
        self.x = x
        self.y = y

    def display_data(self):
        plt.scatter(self.x[:, 0], self.x[:, 1])
        plt.show()
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap="brg")
        plt.show()