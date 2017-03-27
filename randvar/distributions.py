from randvar import get_default_viability, RandVar
import math

def const(val, viability=None):
	"""
	Returns a random variable that takes the value 'val' with probability 1.0.
	"""

	if viability is None:
		viability = get_default_viability()
	return RandVar({val: 1.0}, viability=viability)

def uniform(iterable, viability=None):
	"""
	Returns a random variable that takes each value in 'iterable' with equal
	probability.
	"""

	if viability is None:
		viability = get_default_viability()
	obj = tuple(obj)
	n = len(obj)
	return RandVar({val: 1/n for val in obj}, viability=viability)

def poisson(n, expectation=1.0, viability=None, stretch=False):
	"""
	Returns a random variable following a Poisson distribution with expected
	value 'expectation', truncated so that values greater than 'n' are not
	possible.

	If 'stretch' is true, then the probabilities are all scaled equally so that
	they add up to 1.0. Otherwise, the remaining probability is added on to the
	probability for 'n'.
	"""

	if viability is None:
		viability = get_default_viability()
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
