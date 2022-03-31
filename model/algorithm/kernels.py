import numpy as np


def no_kernel(x, y: np.ndarray):
    return x @ y.T
