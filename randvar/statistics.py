from copy import deepcopy
import math
import functools
import operator

from randvar import rand_apply


def mean(var, p=1.0):
    """
    Returns the generalized `p`-mean of the random variable `var`.
    """

    if p == float("inf"):
        return max(val for val in var._dist)
    if p == float("-inf"):
        return min(val for val in var._dist)
    if p == 0.0:
        return functools.reduce(operator.mul, list(
            val ** prob for val, prob in var._dist.items()), 1)
    return sum(prob * (val ** p) for val, prob in var._dist.items()) ** (1 / p)


def expected_value(var):
    """
    Returns the expected value of the random variable `var`.

    Internally different from `mean(var, p=1.0)`, but should return essentially
    the same result.
    """

    return sum(val * prob for val, prob in var._dist.items())


def percentile(var, p):
    """
    Returns the `p`th percentile value of the random variable `var`.
    """

    pairs = sorted(((val, prob) for val, prob in var._dist.items()),
                   key=lambda pair: pair[0])
    for val, prob in pairs:
        if p < prob:
            return deepcopy(val)
        p -= prob


def median(var):
    """
    Returns the median value of the random variable `var`. Internally 
    computed as the 50th percentile value.
    """

    return percentile(var, 0.5)


def mode(var, k=1):
    """
    Returns the `k`th most probable value of the random variable `var`.
    """

    pairs = sorted(var._dist.keys(), key=lambda val: var[val], reverse=True)
    return pairs[k - 1]


def variance(var):
    """
    Returns the variance of the random variable `var`.
    """

    def sqr(x): return x * x

    return expected_value(rand_apply(sqr, var)) - sqr(expected_value(var))


def stddev(var):
    """
    Returns the standard deviation of the random variable `var`.
    """

    return math.sqrt(variance(var))
