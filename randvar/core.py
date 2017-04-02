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

	def __init__(self, dist, viability=None):
		"""
		Creates a finite random variable with the distribution 'dist'. If the
		probabilities in dist are less that 'viability' away from 1.0, then a
		'ViabilityError' is raised.
		"""

		if viability is None:
			viability = DEFAULT_VIABILITY
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
	# Helper function for '_it_prod'

	try:
		mine = tuple(next(iterator))
	except StopIteration:
		return ((),)
	else:
		theirs = tuple(_it_prod_help(iterator))
		return map(lambda pair: (pair[0],) + pair[1], itertools.product(mine, theirs))

def _it_prod(iterator):
	# Like 'itertools.product', but takes one argument which is in iterator that
	# generates iterators, and the iterators so generated are multiplied
	# together.

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

	rand_args = []
	for arg in args:
		if isinstance(arg, RandVar):
			rand_args.append(arg)
		else:
			rand_args.append(RandVar({arg: 1.0}))
	rand_kwargs = {}
	for name in kwargs:
		if isinstance(kwargs[name], RandVar):
			rand_kwargs[name] = kwargs[name]
		else:
			rand_kwargs[name] = RandVar({kwargs[name]: 1.0})
	viability = min(itertools.chain([viability],
									(sarg._viability for sarg in rand_args),
									(rand_kwargs[name]._viability for name in rand_kwargs)))
	dist = {}
	for args_items in _it_prod(var._dist.items() for var in rand_args):
		if len(args_items) > 0:
			args_inst, args_prob = tuple(zip(*args_items))
			args_inst = deepcopy(args_inst)
		else:
			args_inst = []
			args_prob = []
		for kwargs_items in _it_prod(kwargs[name]._dist.items() for name in rand_kwargs):
			if len(kwargs_items) > 0:
				kwargs_inst, kwargs_prob = tuple(zip(*kwargs_items))
				kwargs_inst = {name: deepcopy(kwargs_inst[i]) for name in rand_kwargs}
			else:
				kwargs_inst = {}
				kwargs_prob = []
			prob = functools.reduce(operator.mul, itertools.chain([1.0], args_prob, kwargs_prob))
			# prob = functools.reduce(operator.mul,
			# 						itertools.chain([1.0],
			# 										(rand_args[i]._dist[args_inst[i]] for i in range(len(rand_args))))) * \
			# 	   functools.reduce(operator.mul,
			# 	   					itertools.chain([1.0],
			# 	     								(rand_kwargs[name]._dist[kwargs_inst[name]] for name in rand_kwargs)))
			if prob > 0.0:
				val = func(*args_inst, **kwargs_inst)
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