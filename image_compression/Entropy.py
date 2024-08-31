import os
import numpy as np
import math


def entropy(file, z=0, y=0, x=0, dtype=0, n_clusters=None):
    G = np.fromfile(file, dtype=dtype, count=x*y*z)
    im = np.reshape(G, (x * y, z), order="F")

    e = _entropy(im)

    return e


def _entropy(data):
    """Compute the zero-order entropy of the provided data
    """
    values, count = np.unique(data.flatten(), return_counts=True)
    total_sum = sum(count)
    probabilities = (count / total_sum for value, count in zip(values, count))
    return -sum(p * math.log2(p) for p in probabilities)