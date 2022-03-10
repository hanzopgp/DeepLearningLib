from Core import *

class Linear(Module):
	def __init__(self, size_in, size_out):
		super().__init__()
		self._parameters = np.random.rand(size_in, size_out)
		self._bias = np.random.rand(size_out)
		self.zero_grad()

	def zero_grad(self):
		## Annule gradient
		self._gradient = np.zeros_like(self._parameters)

	def forward(self, X):
		## Calcule la passe forward
		return np.dot(X.T, self._parameters) + self._bias

	def backward_update_gradient(self, input, delta):
		## Met a jour la valeur du gradient
		raise NotImplementedError()
		

	def backward_delta(self, input, delta):
		## Calcul la derivee de l'erreur
		raise NotImplementedError()