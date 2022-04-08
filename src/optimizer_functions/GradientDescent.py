from Core import *
from global_imports import *
from global_variables import *


class GradientDescent(Optimizer):
	def __init__(self, net, loss_function, learning_rate, decay):
		super().__init__()
		self._net = net
		self._net._network.append(loss_function)
		self._decay = decay
		self._learning_rate = learning_rate

	def step(self, X, y, n_epochs, verbose, early_stopping):
		for cpt_epoch in range(n_epochs):
			self._net.forward(X, y)
			self._net.backward()
			self._net.update_parameters(self._learning_rate)
			self._net.update_stats()

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