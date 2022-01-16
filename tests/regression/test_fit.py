from typing import Tuple
from scipy import sparse as sps
import numpy as np
from ..util import FMWeights

from myfm import MyFMGibbsRegressor

import pytest


@pytest.mark.parametrize("alpha_inv", [0.1, 10.0, 1, 0.3, 3])
def test_middle_reg(
    alpha_inv: float,
    middle_data: Tuple[sps.csr_matrix, np.ndarray],
    stub_weight: FMWeights,
) -> None:
    rns = np.random.RandomState(0)
    X, score = middle_data
    y = score + alpha_inv * rns.normal(0, 1, size=score.shape)

    fm = MyFMGibbsRegressor(4).fit(
        X, y, X_test=X, y_test=y, n_iter=50  # , n_kept_samples=50
    )
    hp_trance = fm.get_hyper_trace()
    last_alphs = hp_trance["alpha"].iloc[-20:].values
    assert np.all(last_alphs > ((1 / alpha_inv ** 2) / 2))
    assert np.all(last_alphs < ((1 / alpha_inv ** 2) * 2))
