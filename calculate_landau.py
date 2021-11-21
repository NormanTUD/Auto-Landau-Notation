import time
import sys
import sympy as sp
from gplearn.genetic import SymbolicRegressor
import numpy as np
from pprint import pprint

def dier (msg):
    pprint(msg)
    sys.exit(1)

x0 = np.arange(-1, 1, .1)
x1 = np.arange(-1, 1, .1)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1 #true function

input_data = [[1, 1], [2, 2], [3, 5], [1, 2], [1, 0], [0, 1]]
results = []

i = 0
for dataset in input_data:
    print("Trying dataset " + str(i) + " of " + str(len(input_data)))
    start = time.time()
    time.sleep(dataset[0] + dataset[1])
    end = time.time()
    results.append(end - start)
    i = i + 1

X_train = np.array(input_data)
y_train = np.array(results)

est_gp = SymbolicRegressor(population_size=5000, #the number of programs in each generation
                           generations=10, stopping_criteria=0.01, #The required metric value required in order to stop evolution early.
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, #0.05, The probability of performing hoist mutation on a tournament winner. Hoist mutation takes the winner of a tournament and selects a random subtree from it. A random subtree of that subtree is then selected and this is ‘hoisted’ into the original subtrees location to form an offspring in the next generation. This method helps to control bloat.
                           p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.1, random_state=0)
est_gp.fit(X_train, y_train)

best = est_gp._program

print(best)

x = sp.symbols('x')
y = sp.symbols("y")

locals = {
    "add": sp.Add,
    "mul": sp.Mul,
    "sub": sp.Lambda((x, y), x - y),
    "div": sp.Lambda((x, y), x/y),
    "X0": sp.symbols("SleepA"),
    "X1": sp.symbols("SleepB")
}

print(sp.sympify(str(best), locals=locals))
