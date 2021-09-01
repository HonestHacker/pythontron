import numpy as np
from . import functions

__author__ = "Makar Kuznetsov"

class Network:
	def __init__(self, neurons, primers, answers, problem, * , seed=1, iterations=1000000, activation=functions.sigmoid):
		self.primers = primers
		self.answers = answers
		self.problem = problem
		self.syn_w = 2 * np.random.random((neurons, 1)) - 1
		self.iterations = iterations
		self.activation=activation
		np.random.seed(seed)
	def start_learn(self):
		for _ in range(self.iterations):
			input_layer = self.primers
			outputs = self.activation(np.dot(input_layer, self.syn_w))
			err = self.answers - outputs
			adjuments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
			self.syn_w += adjuments
			return self.syn_w
	def solve_problem(self):
		answer = self.activation(np.dot(self.problem, self.syn_w))
		return answer
