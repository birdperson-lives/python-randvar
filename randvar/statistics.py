from collections import namedtuple
import math

from randvar import rand_apply


def mean(var, p=1):
    """
    Returns the generalized `p`-mean of the random variable `var`.
    """

    if p == float("inf"):
        return max(val for val in var)
    if p == float("-inf"):
        return min(val for val in var)
    if p == 0:
        return math.exp(sum(weight * math.log(val)
                            for val, weight in var._dist.items()) /
                        var._weight_sum)
    return (sum(weight * (val ** p) for val, weight in var._dist.items()) /
            var._weight_sum) ** (1 / p)


def expected_value(var):
    """
    Returns the expected value of the random variable `var`.

    Internally different from `mean(var, p=1.0)`, but should return essentially
    the same result.
    """

    return sum(val * weight for val, weight in var._dist.items()) / \
           var._weight_sum


def percentile(var, p):
    """
    Returns the `p` percentile value of the random variable `var` (i.e. `0 
    <= p <= 1`).
    """

    ordered_values = sorted(var)
    PercentileNode = namedtuple("PercentileNode", ["value", "lower", "upper"])
    nodes = []
    cum = 0
    for val in ordered_values:
        nodes.append(PercentileNode(value=val,
                                    lower=cum,
                                    upper=cum + var._dist[val]))
        cum += var._dist[val]

    # Binary search
    target = p * var._weight_sum
    bot = 0
    top = len(nodes)
    ind = (bot + top) // 2
    item = nodes[ind]
    while target < item.lower or target >= item.upper:
        if target < item.lower:
            top = ind
        else:
            bot = ind + 1
        ind = (bot + top) // 2
        item = nodes[ind]

    return item.value


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
