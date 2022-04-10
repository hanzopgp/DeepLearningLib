from global_imports import *
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from utils.utils import one_hot

YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
ENDC = "\033[0m"

def binary_classif_score(
	Y_hat: np.ndarray,
	Y: np.ndarray
	):
	predictions = np.argmax(Y_hat, axis=1).reshape(-1, 1)
	return np.sum(predictions == Y) / Y.shape[0]

def multi_classif_score(
	Y_hat: np.ndarray,
	Y: np.ndarray
	):
	oh = not (len(Y.shape) == 1 or Y.shape[1] == 1)	# labels are one-hot encoded?
	Y = np.argmax(Y, axis=1) if oh else Y.reshape(-1)
	predictions = np.argmax(Y_hat, axis=1)
	return np.sum(predictions == Y) / Y.shape[0]

def mse_score(
	Y_hat: np.ndarray,
	Y: np.ndarray
	):
	return np.mean((Y_hat - Y) ** 2)

def run_test(
	test_name: str,						# Name of test for displaying purpose
	X: np.ndarray, Y: np.ndarray,		# Data
	layers: List[Tuple[Linear, str]],	# NN's layers in a list of tuple (Linear, activation_function_name)
	model_kwargs: Dict[str, Any],		# Keyword arguments to pass into NN's constructor
	compile_kwargs: Dict[str, Any],		# Keyword arguments to pass into NN's compile function
	fit_kwargs: Dict[str, Any],			# Keyword arguments to pass into NN's fit function
	train_valid_test=(0.6,0.2,0.2),		# Size of train, validation and test set, must sum to 1
	target_score=0.85,					# Desired NN's score for pass/fail assertion. Can be set to None
	scoring_func=mse_score,				# Function to calculate prediction's score
	scoring_method="lt"					# Comparator to NN's target score to decide whether test passes or fails, "lt" or "gt"
	):
	
	print(f"Testing {YELLOW}{test_name}{ENDC}:")

	r_train, r_valid, r_test = train_valid_test
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=r_test)
	X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=(r_valid/(r_valid+r_train)))

	model = Sequential() if model_kwargs is None else Sequential(**model_kwargs)
	for l, a in layers:
		model.add(layer=l, activation=a)
	model.compile(**compile_kwargs)
	model.fit(X_train, Y_train, X_valid, Y_valid, **fit_kwargs)
	
	score = scoring_func(model.predict(X_test), Y_test)
	print("  Score: %.4f " % score, end='')
	if target_score is None \
		or (scoring_method == "lt" and score <= target_score) \
		or (scoring_method == "gt" and score >= target_score):
		print(f"{GREEN}OK{ENDC}")
		return True
	print(f"{RED}KO{ENDC}")
	return False
	

if __name__ == '__main__':
	np.random.seed(42)
	
	#############################################################################
	print("===== SIMPLE CLASSIFICATION PROBLEM WITH 2 CLASSES =====")
	gen2C = TwoClassGen()
	gen2C.make_2_gaussians(sigma=0.5)
	# gen2C.display_data()

	test_params = {
		"2 Gaussians, BCE loss, GD": ("binary_crossentropy", "gd"),
		"2 Gaussians, BCE loss, SGD": ("binary_crossentropy", "sgd"),
		"2 Gaussians, BCE loss, MGD": ("binary_crossentropy", "mgd"),
		"2 Gaussians, BCE loss, ADAM": ("binary_crossentropy", "adam"),
		"2 Gaussians, Sparse BCE loss, GD": ("sparse_binary_crossentropy", "gd"),
		"2 Gaussians, Sparse BCE loss, SGD": ("sparse_binary_crossentropy", "sgd"),
		"2 Gaussians, Sparse BCE loss, MGD": ("sparse_binary_crossentropy", "mgd"),
		"2 Gaussians, Sparse BCE loss, ADAM": ("sparse_binary_crossentropy", "adam")
	}
	for name in test_params:
		loss, optim = test_params[name]
		if "sparse" not in loss:
			Y = one_hot(gen2C.y, 2)
		else:
			Y = gen2C.y
		run_test(
			test_name=name,
			X=gen2C.x, Y=Y,
			layers=[
				(Linear(2, 4), "tanh"),
				(Linear(4, 2), "sigmoid")
			],
			model_kwargs=None,
			compile_kwargs=dict(
				loss=loss,
				optimizer=optim,
				learning_rate=1e-4,
				metric="accuracy"
			),
			fit_kwargs=dict(
				n_epochs=50,
				verbose=False
			),
			target_score=0.85,
			scoring_func=binary_classif_score,
			scoring_method="gt"
		)

	gen2C.make_4_gaussians(sigma=0.2)
	# gen2C.display_data()
	test_params = {
		"4 Gaussians, BCE loss, GD optim": ("binary_crossentropy", "gd"),
		"4 Gaussians, BCE loss, SGD optim": ("binary_crossentropy", "sgd"),
		"4 Gaussians, BCE loss, MGD optim": ("binary_crossentropy", "mgd"),
		"4 Gaussians, BCE loss, ADAM optim": ("binary_crossentropy", "adam"),
		"4 Gaussians, Sparse BCE loss, GD optim": ("sparse_binary_crossentropy", "gd"),
		"4 Gaussians, Sparse BCE loss, SGD optim": ("sparse_binary_crossentropy", "sgd"),
		"4 Gaussians, Sparse BCE loss, MGD optim": ("sparse_binary_crossentropy", "mgd"),
		"4 Gaussians, Sparse BCE loss, ADAM optim": ("sparse_binary_crossentropy", "adam")
	}
	for name in test_params:
		loss, optim = test_params[name]
		if "sparse" not in loss:
			Y = one_hot(gen2C.y, 2)
		else:
			Y = gen2C.y
		run_test(
			test_name=name,
			X=gen2C.x, Y=Y,
			layers=[
				(Linear(2, 4), "tanh"),
				(Linear(4, 2), "sigmoid")
			],
			model_kwargs=None,
			compile_kwargs=dict(
				loss=loss,
				optimizer=optim,
				learning_rate=1e-2,
				n_batch = 20,
				metric="accuracy"
			),
			fit_kwargs=dict(
				n_epochs=50,
				verbose=False
			),
			target_score=0.85,
			scoring_func=binary_classif_score,
			scoring_method="gt"
		)
	del gen2C

	#############################################################################
	print(end='\n')
	nb_class = 4
	print(f"===== CLASSIFICATION WITH {nb_class} CLASSES =====")
	gen4C = MultiClassGen(nb_class)
	gen4C.make_vertical()
	# gen4C.display_data()
	
	test_params = {
		"Vertical data, CCE, GD optim": ("categorical_crossentropy", "gd"),
		"Vertical data, CCE, SGD optim": ("categorical_crossentropy", "sgd"),
		"Vertical data, CCE, MGD optim": ("categorical_crossentropy", "mgd"),
		"Vertical data, CCE, ADAM optim": ("categorical_crossentropy", "adam"),
		"Vertical data, Sparse CCE, GD optim": ("sparse_categorical_crossentropy", "gd"),
		"Vertical data, Sparse CCE, SGD optim": ("sparse_categorical_crossentropy", "sgd"),
		"Vertical data, Sparse CCE, MGD optim": ("sparse_categorical_crossentropy", "mgd"),
		"Vertical data, Sparse CCE, ADAM optim": ("sparse_categorical_crossentropy", "adam")
	}
	for name in test_params:
		loss, optim = test_params[name]
		if "sparse" not in loss:
			Y = one_hot(gen4C.y, 4)
		else:
			Y = gen4C.y
		run_test(
			test_name=name,
			X=gen4C.x, Y=Y,
			layers=[
				(Linear(2, nb_class * 4), "tanh"),
				(Linear(nb_class * 4, nb_class), "sigmoid")
			],
			model_kwargs=None,
			compile_kwargs=dict(
				loss=loss,
				optimizer=optim,
				learning_rate=0.01,
				metric="accuracy"
			),
			fit_kwargs=dict(
				n_epochs=150,
				verbose=False
			),
			target_score=0.85,
			scoring_func=multi_classif_score,
			scoring_method="gt"
		)
	del gen4C

	#############################################################################
	print(end='\n')
	nb_class = 4
	print("===== REGRESSION PROBLEM =====")
	genCont = ContinuousGen()
	genCont.make_regression()
	# genCont.display_data()
	
	params_optim = [
		"gd",
		"sgd",
		"mgd",
		"adam"
	]
	params_loss = [
		"mse",
		"mae",
		"rmse"
	]

	for optim in params_optim:
		for loss in params_loss:
			name = f"Optimizer {optim}\tLoss function {loss}"
			run_test(
				test_name=name,
				X=genCont.x, Y=genCont.y,
				layers=[
					(Linear(1, 4), "relu"),
					(Linear(4, 1), "linear")
				],
				model_kwargs=None,
				compile_kwargs=dict(
					loss=loss,
					optimizer=optim,
					#learning_rate=8e-4
					learning_rate=1e-4 if optim != "mgd" else 1e-5
				),
				fit_kwargs=dict(
					n_epochs=150,
					verbose=False
				),
				target_score=0.1,
				scoring_func=mse_score,
				scoring_method="lt"
			)
	del genCont
