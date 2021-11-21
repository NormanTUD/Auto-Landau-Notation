import time
from sympy import init_printing
import sys
import sympy as sp
from gplearn.genetic import SymbolicRegressor
import numpy as np
from pprint import pprint
import sympy.printing as printing
from numpy.random import seed
from numpy.random import randint

def mergeSort(arr):
    if len(arr) > 1:

         # Finding the mid of the array
        mid = len(arr)//2

        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def partition(array, begin, end):
    pivot = begin
    for i in range(begin+1, end+1):
        if array[i] <= array[begin]:
            pivot += 1
            array[i], array[pivot] = array[pivot], array[i]
    array[pivot], array[begin] = array[begin], array[pivot]
    return pivot

def quicksort(array, begin=0, end=None):
    if end is None:
        end = len(array) - 1
    def _quicksort(array, begin, end):
        if begin >= end:
            return
        pivot = partition(array, begin, end)
        _quicksort(array, begin, pivot-1)
        _quicksort(array, pivot+1, end)
    return _quicksort(array, begin, end)

init_printing()

def dier (msg):
    pprint(msg)
    sys.exit(1)

x0 = np.arange(-1, 1, .1)
x1 = np.arange(-1, 1, .1)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1 #true function

input_data = []
for i in range(0, 9000, 10):
    input_data.append([i])
results = []

i = 1
for dataset in input_data:
    print("Trying dataset " + str(i) + " of " + str(len(input_data)))

    values = randint(0, 10, dataset[0])

    start = time.time()
    quicksort(values)
    end = time.time()

    print("Got %.5f" % (end - start))

    results.append((end - start))
    i = i + 1

X_train = np.array(input_data)
y_train = np.array(results)

function_set = [
    'add', 'sub', 'mul', 'log'
    #'sqrt', 'log'#, 'abs', 'neg', 'inv', 'div'
    #'max', 'min'
]

est_gp = SymbolicRegressor(
    population_size=5000, #the number of programs in each generation
    generations=50,
    const_range=None,
    stopping_criteria=0.01, #The required metric value required in order to stop evolution early.
    function_set=function_set,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.1, 
#0.05, The probability of performing hoist mutation on a tournament winner. Hoist mutation takes the winner of a tournament and selects a random subtree from it. A random subtree of that subtree is then selected and this is ‘hoisted’ into the original subtrees location to form an offspring in the next generation. This method helps to control bloat.
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.1
)
est_gp.fit(X_train, y_train)

best = est_gp._program

x = sp.symbols('x')
y = sp.symbols("y")

locals = {
    "add": sp.Add,
    "mul": sp.Mul,
    "sub": sp.Lambda((x, y), x - y),
    "div": sp.Lambda((x, y), x/y),
    "X0": sp.symbols("x"),
}

print("f(x) = " + str(sp.sympify(str(best), locals=locals)))

