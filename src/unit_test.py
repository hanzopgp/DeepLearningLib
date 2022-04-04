from global_imports import *
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
ENDC = "\033[0m"

def classif_score(
	Y_hat: np.ndarray,
	Y: np.ndarray
	):
	predictions = np.argmax(Y_hat, axis=1).reshape(-1, 1)
	print(predictions.shape, Y.shape)
	return np.sum(predictions == Y) / Y.shape[0]

def mse_score(
	Y_hat: np.ndarray,
	Y: np.ndarray
	):
	return np.mean((Y_hat - Y) ** 2, axis=0)

def run_test(
	test_name: str,						# Name of test for displaying purpose
	X: np.ndarray,						# Input
	Y: np.ndarray,						# Output
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
	
	gen2C = TwoClassGen()
	gen2C.make_2_gaussians(sigma=0.5)
	gen2C.display_data()
	run_test(
		test_name="Simple binary test",
		X=gen2C.x, Y=gen2C.y,
		layers=[
			(Linear(2, 4), "tanh")
		],
		model_kwargs=None,
		compile_kwargs={
			"loss": "sparse_categorical_crossentropy",
			"optimizer": "SGD",
			"learning_rate": 1e-3,
			"metric": "accuracy"
		},
		fit_kwargs=dict(
			n_epochs=10,
			verbose=False
		),
		target_score=0.9,
		scoring_func=classif_score,
		scoring_method="gt"
	)

	# gen2C.make_checker_board()
	# gen2C.display_data()
	# gen2C.make_4_gaussians(sigma=0.5)
	# gen2C.display_data()

	# gen4C = MultiClassGen(nb_class=4)
	# gen4C.make_vertical()
	# gen4C.display_data()
	# gen4C.make_spiral()
	# gen4C.display_data()

	# cont = ContinuousGen(sigma=0.2)
	# cont.make_sinus(freq=3, ampli=3)
	# cont.display_data()
	# cont.make_regression(slope=5, affine=10)
	# cont.display_data()
