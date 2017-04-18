import math

from randvar import RandomVariable


def _binom(n, k):
    return math.exp(math.lgamma(n) - math.lgamma(k) - math.lgamma(n - k))


def _beta(x, y):
    return math.exp(math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y))


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


def bernoulli(p):
    return RandomVariable({0: 1 - p, 1: p})


def binomial(n, p):
    dist = {k: _binom(n, k) * math.exp(k * math.lgamma(p) +
                                       (n - k) * math.lgamma(1 - p))
            for k in range(n + 1)}
    return RandomVariable(dist)


def beta_binomial(n, alpha, beta):
    dist = {k: _binom(n, k) * _beta(k + alpha, n - k + beta) /
               _beta(alpha, beta) for k in range(n + 1)}
    return RandomVariable(dist)


def hypergeometric(size, good, draws):
    dist = {k: _binom(good, k) * _binom(size - good, draws - k) / _binom(
        size, draws) for k in range(max(0, draws + good - size),
                                    min(good, draws) + 1)}
    return RandomVariable(dist)


def geometric_trunc(p, top):
    dist = {}
    total = 1 / p
    for i in range(1, top + 1):
        dist[i] = math.exp((i - 1) * math.log(1 - p))
        total -= dist[i]
    dist[top] += total
    return RandomVariable(dist)


def geometric_stretch(p, top):
    dist = {i: math.exp((i - 1) * math.log(1 - p)) for i in range(1, top + 1)}
    return RandomVariable(dist)


def poisson_trunc(expectation, top):
    """
    Returns a random variable following a Poisson distribution with expected
    value `expectation`, truncated so that values greater than `n` are not 
    possible.

    The probability that the value exceeds `n` is added onto the probability
    for `n`.
    """

    dist = {}
    total = 1
    for i in range(top + 1):
        dist[i] = math.exp(i * math.log(expectation) - expectation -
                           math.lgamma(i + 1))
        total -= dist[i]
    dist[top] += total
    return RandomVariable(dist)


def poisson_stretch(expectation, top):
    """
    As `poisson_trunc`, but instead of adding the leftover probability to the
    `n`, all probabilities are scaled uniformly to make the total probability
    `1`.
    """

    dist = {}
    for i in range(top + 1):
        dist[i] = math.exp(i * math.log(expectation) - expectation -
                           math.lgamma(i + 1))
    return RandomVariable(dist)


def negative_binomial_trunc(fails, p, top):
    dist = {}
    total = 1
    for i in range(top + 1):
        dist[i] = _binom(i + fails - 1, i) * \
                  math.exp(i * math.log(p) + fails * math.log(1 - p))
        total -= dist[i]
    dist[top] += total
    return RandomVariable(dist)


def negative_binomial_stretch(fails, p, top):
    dist = {}
    for i in range(top + 1):
        dist[i] = _binom(i + fails - 1, i) * \
                  math.exp(i * math.log(p) + fails * math.log(1 - p))
    return RandomVariable(dist)
