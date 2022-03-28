import numpy as np


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,\
 without any for loop.
The three arrays must have compatible shapes.
Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta is an empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
Raises:
    This function should not raise any Exception.
"""
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape 2 * 1.")
        return None
    if x.shape != y.shape:
        print('x and y must have the same shape.')
        return None
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return x.T @ (x @ theta - y) / x.shape[0]
