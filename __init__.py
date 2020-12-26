import numpy as np
from functions import sigmoid

class Network:
	def __init__(self, neurons, primers, answers, problem, * , seed=1, iterations=1000000):
		self.primers = primers
		self.answers = answers
		self.problem = problem
		self.syn_w = 2 * np.random.random((neurons, 1)) - 1
		self.iterations = iterations
		np.random.seed(seed)
	def learn_start(self):
		for i in range(self.iterations):
			input_layer = self.primers
			outputs = sigmoid(np.dot(input_layer, self.syn_w))
			err = self.answers - outputs
			adjuments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
			self.syn_w += adjuments
			return self.syn_w
	def solve_problem(self):
		answer = sigmoid(np.dot(self.problem, self.syn_w))
		return answer
