from typing import NamedTuple

import numpy as np
import scipy.sparse as sps

N_FEATURES = 3
N_LATENT = 4


class FMWeights(NamedTuple):
    global_bias: float
    weight: np.ndarray
    factors: np.ndarray


def prediction(X: sps.csr_matrix, weight: FMWeights) -> np.ndarray:
    X2 = X.copy()
    X2.data[:] = X2.data**2
    result = np.zeros(X.shape[0], dtype=np.float64)
    result[:] = weight.global_bias
    result += X.dot(weight.weight)
    w2 = (weight.factors**2).sum(axis=0)
    Xw = X.dot(weight.factors.T)
    result += ((Xw**2).sum(axis=1) - (X2.dot(w2))) * 0.5
    return result
