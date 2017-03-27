from copy import deepcopy
from random import random
import operator
import itertools
import functools

DEFAULT_VIABILITY = 0.00001

class ViabilityError(Exception):
	"""
	Exception raised when floating-point rounding has degraded values beyond
	use.

	Attributes:
		error -- the deviation of the calculated value from what it should be
	"""

	def __init__(self, error):
		self.error = error

class RandVar:
	"""
	A random variable with finite domain.
	"""

	def __init__(self, dist, viability=DEFAULT_VIABILITY):
		"""
		Creates a finite random variable with the distribution 'dist'. If the
		probabilities in dist are less that 'viability' away from 1.0, then a
		'ViabilityError' is raised.
		"""

		for prob in dist.values():
			assert 0.0 <= prob and prob <= 1.0
		err = abs(sum(prob for prob in dist.values()) - 1.0)
		if err > viability:
			raise ViabilityError(err)
		self._viability = viability
		self._dist = deepcopy(dist)

	def __getitem__(self, val):
		"""
		Gets the probability of 'val'. If 'val' is not in the distribution, a
		probability of 0.0 is returned.
		"""

		if val in self._dist:
			return self._dist[val]
		else:
			return 0.0

	def sample(self, size=1):
		"""
		Returns a random sample of 'size' elements from the distribution.
		"""

		if size == 1:
			x = random()
			for val,prob in self._dist.items():
				if x < prob:
					return [deepcopy(val)]
				x -= prob
		else:
			return [self.sample()[0] for _ in range(size)]


def get_default_viability():
	"""
	Returns the default viability for new random variables.
	"""

	return DEFAULT_VIABILITY

def set_default_viability(value):
	"""
	Changes the default viability of new random variables to 'value'.
	"""

	global DEFAULT_VIABILITY
	assert value >= 0.0
	DEFAULT_VIABILITY = value

def _it_prod_help(iterator):
	try:
		mine = tuple(next(iterator))
	except StopIteration:
		return ((),)
	else:
		theirs = tuple(_it_prod_help(iterator))
		return map(lambda pair: (pair[0],) + pair[1], itertools.product(mine, theirs))

def _it_prod(iterator):
	return tuple(_it_prod_help(iterator))

def rand_apply(func, *args, viability=1.0, **kwargs):
	"""
	Applies a function to random variable arguments, returning a random
	variable representing the distribution of return values from the function.

	All non-random variable arguments to the function are treated as constant
	distributions.

	The viability of the returned variable is the minimum of the viabilities of
	the arguments and the viability keyword argument, if given.
	"""

	sargs = []
	for arg in args:
		if isinstance(arg, RandVar):
			sargs.append(arg)
		else:
			sargs.append(RandVar({arg: 1.0}))
	skwargs = {}
	for name in kwargs:
		if isinstance(kwargs[name], RandVar):
			skwargs[name] = kwargs[name]
		else:
			skwargs[name] = RandVar({kwargs[name]: 1.0})
	viability = min(itertools.chain([viability], (sarg._viability for sarg in sargs), (skwargs[name]._viability for name in skwargs)))
	dist = {}
	for targs in _it_prod(var._dist for var in sargs):
		for tkwargs in _it_prod(kwargs[name]._dist for name in skwargs):
			tkwargs = {name: tkwargs[i] for name in skwargs}
			prob = functools.reduce(operator.mul, itertools.chain([1.0], (sargs[i]._dist[targs[i]] for i in range(len(sargs))))) * functools.reduce(operator.mul, itertools.chain([1.0], (skwargs[name]._dist[tkwargs[name]] for name in skwargs)))
			if prob > 0.0:
				val = func(*targs, **tkwargs)
				if val not in dist:
					dist[val] = prob
				else:
					dist[val] += prob
	return RandVar(dist, viability=viability)

def randomable(func, viability=1.0):
	"""
	A function wrapper so that the functions returns a random variable
	representing the distribution of return values of the body given that all
	the arguments are random variables.

	All non-random variable arguments to the function are treated as having
	constant distributions.

	The viability of the returned variable is the minimum of the viabilities of
	the arguments and the viability keyword argument, if given.
	"""

	return lambda *args, **kwargs: rand_apply(func, *args, viability=viability, **kwargs)