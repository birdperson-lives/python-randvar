import math

from randvar import get_default_viability, RandomVariable


def const(val, viability=None):
    """
    Returns a random variable that takes the value `val` with probability 1.0.
    """

    if viability is None:
        viability = get_default_viability()
    return RandomVariable({val: 1.0}, viability=viability)


def uniform(iterable, viability=None):
    """
    Returns a random variable that takes each value in `iterable` with equal
    probability.
    """

    if viability is None:
        viability = get_default_viability()
    iterable = tuple(iterable)
    n = len(iterable)
    return RandomVariable({val: 1 / n for val in iterable},
                          viability=viability)


def poisson_trunc(n, expectation=1.0, viability=None):
    """
    Returns a random variable following a Poisson distribution with expected
    value `expectation`, truncated so that values greater than `n` are not 
    possible.

    The probability that the value exceeds `n` is added onto the probability
    for `n`.
    """

    if viability is None:
        viability = get_default_viability()
    dist = {}
    total = 1.0
    for i in range(n + 1):
        dist[i] = expectation ** i / math.factorial(i) * math.exp(-expectation)
        total -= dist[i]
    dist[n] += total
    return RandomVariable(dist, viability=viability)


def poisson_stretch(n, expectation=1.0, viability=None):
    """
    As `poisson_trunc`, but instead of adding the leftover probability to the
    `n`, all probabilities are scaled uniformly to make the total probability
    `1.0`.
    """

    if viability is None:
        viability = get_default_viability()
    dist = {}
    total = 1.0
    for i in range(n + 1):
        dist[i] = expectation ** i / math.factorial(i) * math.exp(-expectation)
        total -= dist[i]
    for i in range(n + 1):
        dist[i] /= 1 - total
        if dist[i] > 1:
            dist[i] = 1
    return RandomVariable(dist, viability=viability)
