from Core import *
from global_imports import *
from global_variables import *

from layers.Linear import Linear

# Source: https://arxiv.org/pdf/1412.6980.pdf


class Adam(Optimizer):
	def __init__(self, net, loss_function, learning_rate, decay, b1=0.9, b2=0.999, alpha=1e-3, eps=1e-8):
		super().__init__()
		self._net = net
		self._net._network.append(loss_function)
		## ADAM doesn't mutate the learning rate, its adaptability affects the weights
		## but the learning rate value doesn't change over time. This is why we can keep
		## using a decay with ADAM. The decay changes the value which is the maximum threshold
		self._learning_rate = learning_rate
		self._decay = decay
		## ADAM parameters
		self._mw = 0
		self._vw = 0
		self._mb = 0
		self._vb = 0
		self._b1 = b1
		self._b2 = b2
		self._alpha = alpha
		self._eps = eps

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
		for cpt_epoch in range(1, n_epochs+1):
			## Stochastic gradient descent
			iter = tqdm(range(n)) if TQDM_ACTIVATED else range(n)
			for _ in iter:
				## Classic SGD start
				idx = np.random.choice(n)
				x_element, y_element = X[idx].reshape(1, -1), y[idx].reshape(1, -1)
				self._net.forward(x_element, y_element)
				self._net.backward()
				## With our implementation we can't use update_parameters() from sequential module
				## We need to update the parameters in ADAM optimizer
				for layer in self._net._network:
					try: ## Only apply ADAM on layers with parameters
						## Get layer gradients
						grad_w = layer._gradient
						grad_b = layer._gradient_bias
						## Compute momentums
						self._mw = self._b1 * self._mw + (1 - self._b1) * grad_w
						self._vw = self._b2 * self._vw + (1 - self._b2) * np.power(grad_w, 2)
						self._mb = self._b1 * self._mb + (1 - self._b1) * grad_b
						self._vb = self._b2 * self._vb + (1 - self._b2) * np.power(grad_b, 2)
						## Compute corrections
						mw_hat = self._mw / (1 - self._b1 ** cpt_epoch)
						vw_hat = self._vw / (1 - self._b2 ** cpt_epoch)
						mb_hat = self._mb / (1 - self._b1 ** cpt_epoch)
						vb_hat = self._vb / (1 - self._b2 ** cpt_epoch)
						## Compute update value
						w_update = self._learning_rate * mw_hat / (np.sqrt(np.where(vw_hat>0, vw_hat, 0)) + self._eps)
						b_update = self._learning_rate * mb_hat / (np.sqrt(np.where(vb_hat>0, vb_hat, 0)) + self._eps)
						## Update parameters
						layer._parameters -= w_update
						layer._bias -= b_update
						## Reset gradients
						layer._gradient = np.zeros_like(grad_w)
						layer._gradient_bias = np.zeros_like(grad_b)
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