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

def poisson_trunc(n, expectation=1.0, viability=DEFAULT_VIABILITY, stretch=False):
	"""
	Returns a random variable following a Poisson distribution with expected
	value 'expectation', truncated so that values greater than 'n' are not
	possible.

	If 'stretch' is true, then the probabilities are all scaled equally so that
	they add up to 1.0. Otherwise, the remaining probability is added on to the
	probability for 'n'.
	"""

	dist = {}
	total = 1.0
	for i in range(n+1):
		dist[i] = expectation**i/math.factorial(i)*math.exp(-expectation)
		total -= dist[i]
	if stretch:
		for i in range(n+1):
			dist[i] /= 1-total
	else:
		dist[n] += total
	return RandVar(dist, viability=viability)
