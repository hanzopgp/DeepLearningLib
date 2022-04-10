from pyparsing import alphanums
from Core import *
from global_imports import *
from global_variables import *


class Adam(Optimizer):
	def __init__(self, net, loss_function, learning_rate, decay, b1=0.9, b2=0.999, alpha=1e-3, n=1e-8):
		super().__init__()
		self._net = net
		self._net._network.append(loss_function)
		## ADAM doesn't mutate the learning rate, its adaptability affects the weights
		## but the learning rate value doesn't change over time. This is why we can keep
		## using a decay with ADAM. The decay changes the value which is the maximum threshold
		self._learning_rate = learning_rate
		self._decay = decay
		## ADAM parameters
		self._m = 0
		self._v = 0
		self._b1 = b1
		self._b2 = b2
		self._alpha = alpha
		self._n = n

	def step(self, X, y, n_epochs, verbose, early_stopping):
		## Variables for early stopping
		if early_stopping is not None:
			best_cpt_epoch = 0
			best_model = self._net
			best_valid_loss = np.inf
			best_train_loss = np.inf
			best_valid_acc = 0
			best_train_acc = 0
			cpt_patience = 0

		## Epoch loop : one epoch means we trained on the whole dataset once
		n = X.shape[0]
		for cpt_epoch in range(n_epochs):
			## Stochastic gradient descent
			iter = tqdm(range(n)) if TQDM_ACTIVATED else range(n)
			for _ in iter:
				idx = np.random.choice(n)
				x_element, y_element = X[idx].reshape(1, -1), y[idx].reshape(1, -1)
				self._net.forward(x_element, y_element)
				self._net.backward()
				## With our implementation we can't use update_parameters() from sequential module
				## We need to update the parameters in ADAM optimizer
				for layer in self._net._network:
					try: ## Only apply ADAM on layers with parameters
						## Update weights
						grad_p = layer._gradient
						self._m = self._b1 * self._m + (1 - self._b1) * grad_p
						self._v = self._b2 * self._v + (1 - self._b2) * np.power(grad_p, 2)
						m_hat = self._m / (1 - self._b1)
						v_hat = self._v / (1 - self._b2)
						w_update = self._learning_rate * m_hat / (np.sqrt(v_hat) + self._n)
						layer._parameters -= w_update
						## Update bias
						grad_b = layer._gradient_bias
						self._m = self._b1 * self._m + (1 - self._b1) * grad_b
						self._v = self._b2 * self._v + (1 - self._b2) * np.power(grad_b, 2)
						m_hat = self._m / (1 - self._b1)
						v_hat = self._v / (1 - self._b2)
						b_update = self._learning_rate * m_hat / (np.sqrt(v_hat) + self._n)
						layer._bias -= b_update
					except:
						continue
			
			## Learning rate computation with decay with respect to cpt_epoch
			self._learning_rate  *= (1. / (1. + self._decay * cpt_epoch))

			## Updating the model stats
			train_loss, train_acc, valid_loss, valid_acc = self._net.update_stats()

			## Displaying infos
			if verbose == True: 
				self._net.show_updates(cpt_epoch, self._learning_rate)

			## Early stopping part
			if early_stopping is not None:
				if early_stopping["metric"] == "valid_loss":
					## np.abs() for distance to 0 in case we use BCE loss for example which can give negativ losses
					if np.abs(valid_loss + early_stopping["min_delta"]) < np.abs(best_valid_loss):
						best_cpt_epoch = cpt_epoch
						best_valid_loss = valid_loss
						best_model = self._net
						cpt_patience = 0
					else:
						cpt_patience += 1
						if cpt_patience >= early_stopping["patience"]:
							self._net = best_model
							print("--> Early stopping triggered and best model returned from epoch number", best_cpt_epoch)
							break
				elif early_stopping["metric"] == "train_loss":
					if np.abs(train_loss + early_stopping["min_delta"]) < np.abs(best_train_loss):
						best_cpt_epoch = cpt_epoch
						best_train_loss = train_loss
						best_model = self._net
						cpt_patience = 0
					else:
						cpt_patience += 1
						if cpt_patience >= early_stopping["patience"]:
							self._net = best_model
							print("--> Early stopping triggered and best model returned from epoch number", best_cpt_epoch)
							break
				elif early_stopping["metric"] == "valid_accuracy":
					if (valid_acc - early_stopping["min_delta"]) > best_valid_acc:
						best_cpt_epoch = cpt_epoch
						best_valid_acc = valid_acc
						best_model = self._net
						cpt_patience = 0
					else:
						cpt_patience += 1
						if cpt_patience >= early_stopping["patience"]:
							self._net = best_model
							print("--> Early stopping triggered and best model returned from epoch number", best_cpt_epoch)
							break
				elif early_stopping["metric"] == "train_accuracy":
					if (train_acc - early_stopping["min_delta"]) > best_train_acc:
						best_cpt_epoch = cpt_epoch
						best_train_acc = train_acc
						best_model = self._net
						cpt_patience = 0
					else:
						cpt_patience += 1
						if cpt_patience >= early_stopping["patience"]:
							self._net = best_model
							print("--> Early stopping triggered and best model saved from epoch number", best_cpt_epoch)
							break

# @jit(nopython=True, parallel=True, fastmath=True)
# def _adam(network, m, v, b1, b2, learning_rate, n):
# 	for layer in network:
# 		try: ## Only apply ADAM on layers with parameters
# 			grad_p = layer._gradient
# 			m = b1 * m + (1 - b1) * grad_p
# 			v = b2 * v + (1 - b2) * np.power(grad_p, 2)
# 			m_hat = m / (1 - b1)
# 			v_hat = v / (1 - b2)
# 			w_update = learning_rate * m_hat / (np.sqrt(v_hat) + n)
# 			layer._parameters -= w_update
# 		except:
# 			continue