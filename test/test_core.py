from math import factorial, isclose
import itertools
import unittest
from randvar import ViabilityError, RandVar, get_default_viability, set_default_viability, rand_apply, randomable

class TestRandVarMethods(unittest.TestCase):
	# Tests methods of the `RandVar` class

	def test_init(self):
		# Tests `RandVar.__init__`, ensuring that `ViabilityError` is raised
		# under appropriate circumstances and that the distributions is in fact
		# the one passed.

		with self.assertRaises(ViabilityError):
			RandVar({'p': 0.375, 'q': 0.5})
		myvar = RandVar({0: 0.25, 1: 0.5, 2: 0.25})
		self.assertEqual(myvar._dist[0], 0.25)
		self.assertEqual(myvar._dist[1], 0.5)
		self.assertEqual(myvar._dist[2], 0.25)

	def test_getitem(self):
		# Tests `RandVar.__getitem__`, ensuring that it returns the underlying
		# distribution.

		myvar = RandVar({0: 0.25, 1: 0.5, 2: 0.25})
		self.assertEqual(myvar[0], 0.25)
		self.assertEqual(myvar[1], 0.5)
		self.assertEqual(myvar[2], 0.25)

	def test_sample(self):
		# Tests `RandVar.sample`, checking that the sample distribution roughly
		# matches the variable distribution. Has a small chance of failure even
		# if code is working because of inherent randomness.

		myvar = RandVar({0: 0.25, 1: 0.5, 2: 0.25})
		counts = [0,0,0]
		for i in myvar.sample(1000):
			counts[i] += 1
		self.assertLess(abs(counts[0] - 250), 46) # p < 0.001
		self.assertLess(abs(counts[1] - 500), 53) # p < 0.001
		self.assertLess(abs(counts[2] - 250), 46) # p < 0.001


class TestDefaultViabilityFunctions(unittest.TestCase):
	# Tests functions that get and set `DEFAULT_VIABILITY`

	def test_both(self):
		# Simultaneously test `get_default_viability` and
		# `set_default_viability`, making sure that `RandVar.__init__` still
		# raises `ViabilityError` when appropriate.

		old = get_default_viability()
		self.assertEqual(old, 0.00001)
		with self.assertRaises(ViabilityError):
			RandVar({0: 0.99, 1: 0.0099})
		set_default_viability(0.0012)
		self.assertEqual(get_default_viability(), 0.0012)
		RandVar({0: 0.99, 1: 0.0099})
		set_default_viability(old)
		with self.assertRaises(ViabilityError):
			RandVar({0: 0.99, 1: 0.009})


def binom(n, k):
	return factorial(n)/(factorial(k)*factorial(n-k))

class TestRandomWrappers(unittest.TestCase):
	# Tests the functions `rand_apply` and `randomable`

	def test_rand_apply(self):
		# Tests `rand_apply` by generating a binary variable with a 0.4
		# probability of being 0 and comparing the distribution of the sum of
		# eleven independent copies of this variable to the known `n=11`,
		# `p=0.4` binomial distribution.

		p = 0.4
		n = 11
		unitvar = RandVar({0: p, 1: 1-p})
		def my_sum(*args):
			return sum(args)
		sumvar = rand_apply(my_sum, *((unitvar,)*n))
		for k in range(n+1):
			self.assertTrue(isclose(sumvar[k], binom(n,k)*p**k*(1-p)**(n-k), rel_tol=1e-05, abs_tol=1.0))

	def test_randomable(self):
		# Tests `randomable` by generating a variable with a 0.3 probability of
		# being 1 and a 0.7 probability of being -1. Creates a randomable
		# function that multiplies its arguments together. Compares the product
		# of 8 copies of the random variable against the known result
		# distribution.

		p = 0.3
		n = 8
		unitvar = RandVar({1: p, -1: 1-p})
		@randomable
		def prod(*args):
			return itertools.accumulate((1,) + args)
		prodvar = prod(*((unitvar,)*n))
		self.assertTrue(isclose(prodvar[1], 0.50032768, rel_tol=1e-05, abs_tol=1.0))
		self.assertTrue(isclose(prodvar[-1], 0.49967232, rel_tol=1e-05, abs_tol=1.0))


if __name__ == "__main__":
	unittest.main()
