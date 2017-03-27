from core import DEFAULT_VIABILITY, RandVar
import math

def const(val, viability=DEFAULT_VIABILITY):
	"""
	Returns a random variable that takes the value 'val' with probability 1.0.
	"""

	return RandVar({val: 1.0}, viability=viability)

def uniform(iterable, viability=DEFAULT_VIABILITY):
	"""
	Returns a random variable that takes each value in 'iterable' with equal
	probability.
	"""

	obj = tuple(obj)
	n = len(obj)
	return RandVar({val: 1/n for val in obj}, viability=viability)

def poisson_trunc(n, expectation=1.0, viability=DEFAULT_VIABILITY):
	"""
	Returns a random variable following a Poisson distribution with expected
	value 'expectation', truncated so that all values greater than 'n' are
	reduced to n.
	"""

	dist = {}
	total = 1.0
	for i in range(n):
		dist[i] = expectation**i/math.factorial(i)*math.exp(-expectation)
		total -= dist[i]
	dist[n] = total
	return RandVar(dist, viability=viability)