import unittest
from randvar import ViabilityError, RandVar, get_default_viability, set_default_viability, rand_apply, randomable

class TestRandVarMethods(unittest.TestCase):
	# Tests methods of the 'RandVar' class

	def test_init(self):
		# Tests 'RandVar.__init__', ensuring that 'ViabilityError' is raised
		# under appropriate circumstances and that the distributions is in fact
		# the one passed.

		with self.assertRaises(ViabilityError):
			RandVar({'p': 0.375, 'q': 0.5})
		myvar = RandVar({0: 0.25, 1: 0.5, 2: 0.25})
		self.assertEqual(myvar._dist[0], 0.25)
		self.assertEqual(myvar._dist[1], 0.5)
		self.assertEqual(myvar._dist[2], 0.25)

	def test_getitem(self):
		# Tests 'RandVar.__getitem__', ensuring that it returns the underlying
		# distribution.

		myvar = RandVar({0: 0.25, 1: 0.5, 2: 0.25})
		self.assertEqual(myvar[0], 0.25)
		self.assertEqual(myvar[1], 0.5)
		self.assertEqual(myvar[2], 0.25)

	def test_sample(self):
		# Tests 'RandVar.sample', checking that the sample distribution roughly
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
	# Tests functions that get and set 'DEFAULT_VIABILITY'

	def test_both(self):
		# Simultaneously test 'get_default_viability' and
		# 'set_default_viability', making sure that 'RandVar.__init__' still
		# raises 'ViabilityError' when appropriate.

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


if __name__ == "__main__":
	unittest.main()
