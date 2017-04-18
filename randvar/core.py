from copy import deepcopy
from random import random
from collections import namedtuple
import operator
import itertools
import functools

DEFAULT_VIABILITY = 0.00001


class InviableRandomVariableError(Exception):
    """
    Exception raised when floating-point rounding has degraded values beyond
    use.

    Attributes:
        `error` -- the deviation of the calculated value from what it should be
    """

    def __init__(self, error):
        self.error = error


class RandomVariable:
    """
    A random variable with finite domain.
    """

    def __init__(self, dist, viability=None):
        """
        Creates a finite random variable with the distribution 'dist'. If 
        the probabilities in dist are less that 'viability' away from 1.0, 
        then a 'InviableRandomVariableError` is raised.
        """

        if viability is None:
            viability = DEFAULT_VIABILITY
        for prob in dist.values():
            assert 0 <= prob <= 1
        err = abs(sum(prob for prob in dist.values()) - 1.0)
        if err > viability:
            raise InviableRandomVariableError(err)
        self._viability = viability
        self._dist = deepcopy(dist)
        self._search = []
        SearchNode = namedtuple("SearchNode", ["value", "lower", "upper"])
        cum = 0
        for val in dist:
            self._search.append(
                SearchNode(value=val, lower=cum, upper=cum + dist[val]))
            cum += dist[val]

    def __getitem__(self, val):
        """
        Gets the probability of `val`. If `val` is not in the distribution, 
        a probability of 0.0 is returned.
        """

        if val in self._dist:
            return self._dist[val]
        else:
            return 0.0

    def __str__(self):
        return "RandomVariable(%s, viability=%s)" % (
            str(self._dist), str(self._viability))

    def __repr__(self):
        return "RandomVariable(%s, viability=%s)" % (
            str(self._dist), str(self._viability))

    def choice(self):
        """
        Returns a random element from the distribution.
        """

        x = random()
        bot = 0
        top = len(self._search)
        ind = (bot + top) // 2
        item = self._search[ind]
        while x < item.lower or x >= item.upper:
            if x < item.lower:
                top = ind
            else:
                bot = ind + 1
            ind = (bot + top) // 2
            item = self._search[ind]
        return deepcopy(item.value)

    def sample(self, size=1):
        """
        Returns a random sample of `size` elements from the distribution.
        """

        return [self.choice() for _ in range(size)]


def get_default_viability():
    """
    Returns the default viability for new random variables.
    """

    return DEFAULT_VIABILITY


def set_default_viability(value):
    """
    Changes the default viability of new random variables to `value`.
    """

    global DEFAULT_VIABILITY
    assert value >= 0.0
    DEFAULT_VIABILITY = value


def rand_apply(func, *args, viability=1.0, **kwargs):
    """
    Applies a function to random variable arguments, returning a random 
    variable representing the distribution of return values from the function.

    All non-random variable arguments to the function are treated as 
    constant distributions.

    The viability of the returned variable is the minimum of the 
    viabilities of the arguments and the viability keyword argument, if given.
    """

    rand_args = []
    for arg in args:
        if isinstance(arg, RandomVariable):
            rand_args.append(arg)
        else:
            rand_args.append(RandomVariable({arg: 1}))
    ordered_names = [name for name in kwargs]
    rand_kwargs = {}
    for name in kwargs:
        if isinstance(kwargs[name], RandomVariable):
            rand_kwargs[name] = kwargs[name]
        else:
            rand_kwargs[name] = RandomVariable({kwargs[name]: 1})
    viability = min(itertools.chain([viability],
                                    (sarg._viability for sarg in rand_args),
                                    (rand_kwargs[name]._viability for name in
                                     rand_kwargs)))
    dist = {}
    for args_items in itertools.product(
            *tuple(var._dist.items() for var in rand_args)):
        if len(args_items) > 0:
            args_inst, args_prob = tuple(zip(*args_items))
            args_inst = deepcopy(args_inst)
        else:
            args_inst = []
            args_prob = []
        for kwargs_items in itertools.product(
                *(tuple(kwargs[name]._dist.items()
                        for name in ordered_names))):
            if len(kwargs_items) > 0:
                kwargs_inst, kwargs_prob = tuple(zip(*kwargs_items))
                kwargs_inst = {name: deepcopy(kwargs_inst[i]) for i, name in
                               enumerate(ordered_names)}
            else:
                kwargs_inst = {}
                kwargs_prob = []
            prob = functools.reduce(operator.mul,
                                    itertools.chain(args_prob, kwargs_prob),
                                    1)
            if prob > 0:
                val = func(*args_inst, **kwargs_inst)
                if val not in dist:
                    dist[val] = prob
                else:
                    dist[val] += prob
    return RandomVariable(dist, viability=viability)


def randomable(func, viability=1.0):
    """
    A function wrapper so that the functions returns a random variable 
    representing the distribution of return values of the body given that 
    all the arguments are random variables.

    All non-random variable arguments to the function are treated as having 
    constant distributions.

    The viability of the returned variable is the minimum of the
    viabilities of the arguments and the viability keyword argument, if given.
    """

    return lambda *args, **kwargs: rand_apply(func, *args, viability=viability,
                                              **kwargs)
