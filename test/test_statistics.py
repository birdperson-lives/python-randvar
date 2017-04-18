import random
import unittest
from math import factorial, isclose, sqrt
from randvar import uniform, rand_apply, mean, expected_value, percentile, \
    median, mode, variance, stddev


class TestStatistics(unittest.TestCase):
    """
    Tests statistics functions. Work in progress.
    """

    def test_mean(self):
        """
        Tests the `mean` function by computing various means.
        """

        for _ in range(10):
            n = random.randint(5, 100)
            myvar = uniform(range(1, n + 1))
            self.assertEqual(mean(myvar, float("-inf")), 1)
            self.assertEqual(mean(myvar, float("inf")), n)
            self.assertTrue(
                isclose(mean(myvar, 0), factorial(n) ** (1 / n), rel_tol=1e-05,
                        abs_tol=1.0))
            self.assertTrue(isclose(mean(myvar, 1), (n + 1) / 2, rel_tol=1e-05,
                                    abs_tol=1.0))
            self.assertTrue(
                isclose(mean(myvar, 2), sqrt((n + 1) * (2 * n + 1) / 6),
                        rel_tol=1e-05, abs_tol=1.0))
            self.assertTrue(isclose(mean(myvar, -1),
                                    n / sum(1 / k for k in range(1, n + 1)),
                                    rel_tol=1e-05,
                                    abs_tol=1.0))

    def test_expected_value(self):
        """
        Tests the `expected_value` function by computing the expected value of
        various distributions.
        """

        for _ in range(10):
            n = random.randint(10, 100)
            myvar = rand_apply(lambda x: x * x, uniform(range(1, n + 1)))
            self.assertTrue(
                isclose(expected_value(myvar), (n + 1) * (2 * n + 1) / 6,
                        rel_tol=1e-05, abs_tol=1.0))

    def test_percentile(self):
        """
        Tests the `percentile` function by computing the `n`th percentile of
        `uniform(range(100))` for `n` in `range(
        128)`.
        """

        my_var = uniform(range(128))
        for n in range(128):
            self.assertEqual(percentile(my_var, n / 128), n)

    def test_median(self):
        """
        Tests the `median` function by finding the medians of various `range`s.
        """

        for _ in range(10):
            n = random.randint(10, 100)
            myvar = uniform(range(2 * n + 1))
            self.assertEqual(median(myvar), n)

    def test_mode(self):
        """
        Tests the `mode` function by finding the `k`-modes of binomial
        distributions.
        """

        for _ in range(10):
            n = random.randint(2, 7)
            # create a balanced binomial distribution for `range(2 * n + 1)`
            myvar = rand_apply(lambda *args: sum(iter(args)),
                               *tuple(uniform(range(2)) for _ in range(2 * n)))
            got = set()
            self.assertEqual(mode(myvar), n)
            got.add(n)
            for k in range(1, n + 1):
                knowns = {n - k, n + k}
                a = mode(myvar, 2 * k)
                b = mode(myvar, 2 * k + 1)
                self.assertTrue(a in knowns)
                self.assertTrue(b in knowns)
                got.add(a)
                got.add(b)
            self.assertEqual(got, set(range(2 * n + 1)))

    def test_variance(self):
        """
        Tests the `variance` function by finding the variances of uniform
        distributions over `range`s.
        """

        for _ in range(10):
            n = random.randint(10, 100)
            myvar = uniform(range(1, n + 1))
            self.assertTrue(
                isclose(variance(myvar), (n - 1) * (n + 1) / 12, rel_tol=1e-05,
                        abs_tol=1.0))

    def test_stddev(self):
        """
        Tests the `stddev` function by finding the standard deviations of
        uniform distributions over `range`s.
        """

        for _ in range(10):
            n = random.randint(10, 100)
            myvar = uniform(range(1, n + 1))
            self.assertTrue(
                isclose(stddev(myvar), sqrt((n - 1) * (n + 1) / 12),
                        rel_tol=1e-05,
                        abs_tol=1.0))


if __name__ == "__main__":
    unittest.main()
