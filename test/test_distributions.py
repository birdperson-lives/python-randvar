import random
import math
import unittest
from randvar import const, uniform, poisson_trunc, poisson_stretch


class TestDistribuitons(unittest.TestCase):
    """
    Tests pre-built distributions.
    """

    def test_const(self):
        """
        Tests the `const` distribution by ensuring that only one value is ever sampled.
        """

        constvar = const(232)
        for x in constvar.sample(1000):
            self.assertEqual(x, 232)

    def test_uniform(self):
        """
        Tests the `uniform` distribution by ensuring that all values are sampled with equal probability.
        """

        for size in range(1, 10):
            uniformvar = uniform(range(size))
            for n in range(size):
                self.assertTrue(math.isclose(uniformvar[n], 1/size, rel_tol=1e-05, abs_tol=1.0))

    def test_poisson_trunc(self):
        """
        Tests the `poisson_trunc` distribution by comparing to the known distribution.
        """

        for n in range(0, 10):
            expect = 5 * random.random()
            poissonvar = poisson_trunc(n, expect)
            for k in range(n):
                self.assertTrue(math.isclose(poissonvar[k], expect**k/math.factorial(k)*math.exp(-expect),
                                             rel_tol=1e-05, abs_tol=1.0))

    def test_poisson_stretch(self):
        """
        Tests the `poisson_stretch` distribution by comparing to the known distribution.
        """

        for n in range(0, 10):
            expect = 5 * random.random()
            poissonvar = poisson_stretch(n, expect)
            for k in range(1, n+1):
                self.assertTrue(math.isclose(poissonvar[k]/poissonvar[0], expect**k/math.factorial((k)), rel_tol=1e-05,
                                             abs_tol=1.0))


if __name__ == "__main__":
    unittest.main()
