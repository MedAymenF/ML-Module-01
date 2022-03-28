import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array using\
 the min-max standardization.
Args:
    x: has to be an numpy.array, a vector.
Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
Raises:
    This function shouldn’t raise any Exception.
"""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a vector.")
        return None
    min = x.min(axis=0)
    range = x.max(axis=0) - min
    return (x - min) / range
