from typing import List, Tuple

import numpy as np
import pytest
import scipy.sparse as sps

from .test_utils import FMWeights, prediction

N_FEATURES = 3
N_LATENT = 4


@pytest.fixture
def stub_weight() -> FMWeights:
    weights = FMWeights(
        -3.0,
        np.asfarray([1.0, 2.0, -1.0]),
        np.asfarray(
            [[1.0, -1.0, 0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0, -1.0]]
        ),
    )
    return weights


def create_data(
    n_train: int, stub_weight: FMWeights
) -> Tuple[sps.csr_matrix, np.ndarray]:
    rns = np.random.RandomState(0)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for row in range(n_train):
        indices = np.where(rns.random(N_FEATURES) > 0.5)[0]
        for ind in indices:
            rows.append(row)
            cols.append(ind)
            data.append(float(rns.choice([-2, -1, 1, 2])))
    X = sps.csr_matrix((data, (rows, cols)))
    p = prediction(X, weight=stub_weight)
    return X, p


@pytest.fixture
def middle_data(stub_weight: FMWeights) -> Tuple[sps.csr_matrix, np.ndarray]:
    return create_data(1000, stub_weight)
