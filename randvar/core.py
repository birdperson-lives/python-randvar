from copy import deepcopy
from random import random
from collections import namedtuple
import operator
import itertools
import functools

DEFAULT_VIABILITY = 0.00001

class ZeroDistributionError(Exception):
    def __init__(self):
        pass


class RandomVariable:
    """
    A random variable with finite domain.
    """

    def __init__(self, dist):
        """
        Creates a finite random variable with the distribution `dist`, 
        which should be dictionary with any keys to positive numeric 
        weights. The weights need not sum to 1. Rather, each `key` has a 
        probability of `dist[key] / total`, where `total` is the sum of all 
        the weights in `dist`.
        
        Hereafter, the keys of the distribution dictionary are referred to 
        as "values" and the values of the distribution dictionary are referred
        to as "weights". The numbers `dist[val] / total` as above are 
        referred to as "probabilities".
        """

        # Verify the distribution is valid and initialize
        for prob in dist.values():
            assert 0 <= prob
        self._weight_sum = sum(dist.values())
        if self._weight_sum == 0:
            raise ZeroDistributionError()
        self._dist = {val: prob for val, prob in dist.items()}

        # Remove all items from distribution with zero probability (
        # equivalently, zero weight)
        zeros = set()
        for val, weight in self._dist.items():
            if weight == 0:
                zeros.add(val)
        for val in zeros:
            del self._dist[val]

        # Create a search tree for `self.choice()`
        self._search = []
        SearchNode = namedtuple("SearchNode", ["value", "lower", "upper"])
        cum = 0
        for val in dist:
            self._search.append(
                SearchNode(value=val, lower=cum, upper=cum + dist[val]))
            cum += dist[val]

    def __len__(self):
        """
        Returns the number of values in the distribution (and all should have
        non-zero probability).
        """

        return len(self._dist)

    def __getitem__(self, val):
        """
        Returns the probability of `val`. If `val` is not in the distribution, 
        a probability of 0 is returned.
        """

        if val in self._dist:
            return self._dist[val] / self._weight_sum
        else:
            return 0

    def __iter__(self):
        """
        Return an iterator over the values in the distribution.
        """

        return iter(self._dist)

    def __contains__(self, item):
        """
        Returns `True` if the item has a non-zero probability and `False` 
        otherwise.
        """

        return item in self._dist

    def probs(self):
        """
        Returns an iterator over the probabilities in the distribution.
        """
        return (weight / self._weight_sum for weight in self._dist.values())

    def dist(self):
        """
        Returns an iterator over the `(value, probability)` pairs in the 
        distribution.
        """
        return ((val, weight / self._weight_sum) for val, weight in
                 self._dist.items())

    def __str__(self):
        return "RandomVariable(%s)" % str(self._dist)

    def __repr__(self):
        return "RandomVariable(%s)" % str(self._dist)

    def choice(self):
        """
        Returns a random element from the distribution.
        """

        # Uses binary search
        x = random() * self._weight_sum
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


def rand_apply(func, *args, **kwargs):
    """
    Applies a function to random variable arguments, returning a random 
    variable representing the distribution of return values from the function.

    All non-random variable arguments to the function are treated as 
    constant distributions.
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
                *(tuple(rand_kwargs[name]._dist.items()
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
    return RandomVariable(dist)


def randomable(func, viability=1.0):
    """
    A function wrapper so that the functions returns a random variable 
    representing the distribution of return values of the body given that 
    all the arguments are random variables.

    All non-random variable arguments to the function are treated as having 
    constant distributions.
    """

    return lambda *args, **kwargs: rand_apply(func, *args, **kwargs)
