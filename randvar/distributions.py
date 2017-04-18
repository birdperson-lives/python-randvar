import math

from randvar import RandomVariable


def const(val):
    """
    Returns a random variable that takes the value `val` with probability 1.0.
    """

    return RandomVariable({val: 1})


def uniform(iterable):
    """
    Returns a random variable that takes each value in `iterable` with equal
    probability.
    """

    iterable = tuple(iterable)
    return RandomVariable({val: 1 for val in iterable})


def poisson_trunc(n, expectation=1):
    """
    Returns a random variable following a Poisson distribution with expected
    value `expectation`, truncated so that values greater than `n` are not 
    possible.

    The probability that the value exceeds `n` is added onto the probability
    for `n`.
    """

    dist = {}
    total = 1
    for i in range(n + 1):
        dist[i] = math.exp(i * math.log(expectation) - expectation -
                           math.lgamma(i + 1))
        total -= dist[i]
    dist[n] += total
    return RandomVariable(dist)


def poisson_stretch(n, expectation=1):
    """
    As `poisson_trunc`, but instead of adding the leftover probability to the
    `n`, all probabilities are scaled uniformly to make the total probability
    `1`.
    """

    dist = {}
    for i in range(n + 1):
        dist[i] = math.exp(i * math.log(expectation) - expectation -
                           math.lgamma(i + 1))
    return RandomVariable(dist)
