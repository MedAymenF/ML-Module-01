import numpy as np


def zscore(x):
    """Computes the normalized version of a non-empty numpy.array using\
 the z-score standardization.
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
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    return (x - mu) / sigma
