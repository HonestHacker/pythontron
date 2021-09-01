import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
tanh = lambda x: 2  (1 + np.exp(-2*x)) - 1
relu = lambda x: max(0, x)
unit_step = lambda x: 0 if x != 0 else 1
