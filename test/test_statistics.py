import random
import unittest
from math import isclose
from randvar import uniform, mean, expected_value, percentile, median, mode, variance, stddev


class TestStatistics(unittest.TestCase):
    """
    Tests statistics functions. Work in progress.
    """

    # def test_mean(self):
    #     """
    #     Tests the `mean` function by computing various means.
    #     """
    #
    #     for _ in range(10):
    #         n = random.randint(5, 100)
    #         myvar = uniform(n)
