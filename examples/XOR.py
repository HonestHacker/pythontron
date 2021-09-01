import sys
import numpy as np
sys.path.append('../')
from __init__ import Network

learn_in = np.array([
			[0, 0, 1],
			[1, 1, 1],
			[1, 0, 1],
			[0, 0, 0]
					])
learn_out = np.array([[0, 1, 1, 0]]).T
problem = np.array([0, 1, 0])

nc = Network(3, learn_in, learn_out, problem)
nc.learn_start()

print(nc.solve_problem())
