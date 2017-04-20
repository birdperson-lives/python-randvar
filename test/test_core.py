from math import factorial, isclose
from random import randint
import itertools
import unittest

from randvar import EmptyDistributionError, ZeroDistributionError, \
    NegativeWeightError, RandomVariable, rand_apply, randomable


class TestRandomVariableMethods(unittest.TestCase):
    """
    Tests methods of the `RandomVariable` class
    """

    def test_init(self):
        """
        Tests `RandomVariable.__init__`, ensuring that errors are raised 
        when appropriate and that the distribution is in fact the one passed
        otherwise.
        """

        with self.assertRaises(EmptyDistributionError):
            RandomVariable({})

        with self.assertRaises(ZeroDistributionError):
            RandomVariable({'p': 0, 'q': 0.0})

        with self.assertRaises(NegativeWeightError):
            RandomVariable({None: -1})

        myvar = RandomVariable({0: 1, 1: 2, 2: 1, 3: 0})
        self.assertEqual(myvar._weight_sum, 4)
        self.assertEqual(myvar._dist[0], 1)
        self.assertEqual(myvar._dist[1], 2)
        self.assertEqual(myvar._dist[2], 1)
        with self.assertRaises(KeyError):
            myvar._dist[3]

    def test_len(self):
        """
        Tests `RandomVariable.__len__` with various distributions.
        """

        for n in range(1, 1000):
            myvar = RandomVariable({x: 1 for x in range(n)})
            self.assertEqual(len(myvar), n)

    def test_getitem(self):
        """
        Tests `RandomVariable.__getitem__`, ensuring that it returns the
        underlying distribution.
        """

        myvar = RandomVariable({0: 1, 1: 2, 2: 1})
        self.assertEqual(myvar[0], 1 / 4)
        self.assertEqual(myvar[1], 1 / 2)
        self.assertEqual(myvar[2], 1 / 4)

    def test_iter(self):
        """
        Tests `RandomVariable.__iter__` with various distributions.
        """

        for n in range(1, 1000):
            myvar = RandomVariable({x: 1 for x in range(n)})
            self.assertEqual(set(iter(myvar)), set(range(n)))

    def test_contains(self):
        """
        Tests `RandomVariable.__contains__` with various distributions.
        """

        for n in range(1, 1000):
            myvar = RandomVariable({x: 1 for x in range(n)})
            for _ in range(100):
                x = randint(0, 2 * n)
                if x < n:
                    self.assertIn(x, myvar)
                else:
                    self.assertNotIn(x, myvar)

    def test_probs(self):
        """
        Tests `RandomVariable.probs` with various distributions.
        """

        var1 = RandomVariable({None: 12, "word": 24, 42: 36})
        var2 = RandomVariable({frozenset(): 4, 344.123: 200})
        var3 = RandomVariable({(1, 2, 3): 5, object: 1, "stuff": 1})
        self.assertEqual(set(var1.probs()), {1 / 6, 1 / 3, 1 / 2})
        self.assertEqual(set(var2.probs()), {1 / 51, 50 / 51})
        self.assertEqual(set(var3.probs()), {5 / 7, 1 / 7})

    def test_dist(self):
        """
        Tests `RandomVariable.dist` with various distributions.
        """

        var1 = RandomVariable({None: 12, "word": 24, 42: 36})
        var2 = RandomVariable({frozenset(): 4, 344.123: 200})
        var3 = RandomVariable({(1, 2, 3): 5, object: 1, "stuff": 1})
        self.assertEqual(set(var1.dist()), {(None, 1 / 6),
                                            ("word", 1 / 3),
                                            (42, 1 / 2)})
        self.assertEqual(set(var2.dist()), {(frozenset(), 1 / 51),
                                            (344.123, 50 / 51)})
        self.assertEqual(set(var3.dist()), {((1, 2, 3), 5 / 7),
                                            (object, 1 / 7),
                                            ("stuff", 1 / 7)})

    def test_choice(self):
        """
        Tests `RandomVariable.choice`, checking that the values indeed come
        from the distribution.
        """

        myvar = RandomVariable({"heads": 1, "tails": 1})
        counts = {"heads": 0, "tails": 0}
        for _ in range(1000):
            counts[myvar.choice()] += 1
        self.assertEqual(counts["heads"] + counts["tails"], 1000)
        self.assertLessEqual(abs(counts["heads"] - 500), 51)  # p < 0.001
        self.assertLessEqual(abs(counts["tails"] - 500), 51)  # p < 0.001

    def test_sample(self):
        """
        Tests `RandomVariable.sample`, checking that the sample distribution
        roughly matches the variable distribution. Has a small chance of 
        failure even if code is working because of inherent randomness.
        """

        myvar = RandomVariable({0: 0.25, 1: 0.5, 2: 0.25})
        counts = [0, 0, 0]
        for i in myvar.sample(1000):
            counts[i] += 1
        self.assertLess(abs(counts[0] - 250), 46)  # p < 0.001
        self.assertLess(abs(counts[1] - 500), 53)  # p < 0.001
        self.assertLess(abs(counts[2] - 250), 46)  # p < 0.001


def binom(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


class TestRandomWrappers(unittest.TestCase):
    """
    Tests the functions `rand_apply` and `randomable`
    """

    def test_rand_apply(self):
        """
        Tests `rand_apply` by generating a binary variable with a 0.4
        probability of being 0 and comparing the distribution of the sum of 
        eleven independent copies of this variable to the known `n=11`,
        `p=0.4` binomial  distribution.
        """

        p = 0.4
        n = 11
        unitvar = RandomVariable({0: p, 1: 1 - p})

        def mysum(*args):
            return sum(args)

        sumvar = rand_apply(mysum, *((unitvar,) * n))
        for k in range(n + 1):
            self.assertTrue(
                isclose(sumvar[k], binom(n, k) * p ** k * (1 - p) ** (n - k),
                        rel_tol=1e-05, abs_tol=1.0))

        def mydet(a, b, c, d):
            return a * d - b * c

        myvar = RandomVariable({-1: 0.25, 0: 0.5, 1: 0.25})
        detvar = rand_apply(mydet, myvar, myvar, c=myvar, d=myvar)
        self.assertTrue(isclose(detvar[-2], 1 / 64, rel_tol=1e-05,
                                abs_tol=1.0))
        self.assertTrue(isclose(detvar[-1], 3 / 16, rel_tol=1e-05,
                                abs_tol=1.0))
        self.assertTrue(isclose(detvar[0], 19 / 32, rel_tol=1e-05,
                                abs_tol=1.0))
        self.assertTrue(isclose(detvar[1], 3 / 16, rel_tol=1e-05, abs_tol=1.0))
        self.assertTrue(isclose(detvar[2], 1 / 64, rel_tol=1e-05, abs_tol=1.0))

    def test_randomable(self):
        """
        Tests `randomable` by generating a variable with a 0.3 probability 
        of being 1 and a 0.7 probability of being -1. Creates a `randomable`
        function that multiplies its arguments together. Compares the product
        of 8 copies of the random variable against the known result 
        distribution.
        """

        p = 0.3
        n = 8
        unitvar = RandomVariable({1: p, -1: 1 - p})

        @randomable
        def prod(*args):
            return itertools.accumulate((1,) + args)

        prodvar = prod(*((unitvar,) * n))
        self.assertTrue(
            isclose(prodvar[1], 0.50032768, rel_tol=1e-05, abs_tol=1.0))
        self.assertTrue(
            isclose(prodvar[-1], 0.49967232, rel_tol=1e-05, abs_tol=1.0))

        @randomable
        def mydet(a, b, c, d):
            return a * d - b * c

        myvar = RandomVariable({-1: 0.25, 0: 0.5, 1: 0.25})
        detvar = mydet(myvar, myvar, c=myvar, d=myvar)
        self.assertTrue(isclose(detvar[-2], 1 / 64, rel_tol=1e-05,
                                abs_tol=1.0))
        self.assertTrue(isclose(detvar[-1], 3 / 16, rel_tol=1e-05,
                                abs_tol=1.0))
        self.assertTrue(isclose(detvar[0], 19 / 32, rel_tol=1e-05,
                                abs_tol=1.0))
        self.assertTrue(isclose(detvar[1], 3 / 16, rel_tol=1e-05, abs_tol=1.0))
        self.assertTrue(isclose(detvar[2], 1 / 64, rel_tol=1e-05, abs_tol=1.0))


if __name__ == "__main__":
    unittest.main()
