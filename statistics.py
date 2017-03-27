def expected_value(var):
	"""
	Returns the expected value of the random variable 'var'.
	"""

	return sum(val*prob for val,prob in var._dist.items())
