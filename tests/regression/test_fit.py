from typing import Optional, Tuple

import numpy as np
import pytest
from scipy import sparse as sps

from myfm import MyFMGibbsRegressor, VariationalFMRegressor
from myfm.base import DenseArray
from myfm.utils.callbacks import RegressionCallback

from ..test_utils import FMWeights


def assert_unwrap(x: Optional[DenseArray]) -> DenseArray:
    assert x is not None
    return x


@pytest.mark.parametrize("alpha_inv", [0.3, 1.0, 3])
def test_middle_reg(
    alpha_inv: float,
    middle_data: Tuple[sps.csr_matrix, np.ndarray],
    stub_weight: FMWeights,
) -> None:
    rns = np.random.RandomState(0)
    X, score = middle_data
    y = score + alpha_inv * rns.normal(0, 1, size=score.shape)

    callback = RegressionCallback(100, X_test=X, y_test=y)

    fm_init = MyFMGibbsRegressor(3)
    assert fm_init.w0_samples is None
    assert fm_init.w_samples is None
    assert fm_init.V_samples is None
    fm = fm_init.fit(
        X, y, X_test=X, y_test=y, n_iter=100, n_kept_samples=100, callback=callback
    )

    np.testing.assert_allclose(fm.predict(X), callback.predictions / 100)
    vfm = VariationalFMRegressor(3).fit(X, y, X_test=X, y_test=y, n_iter=50)
    vfm_weights = vfm.predictor_.weights()
    hp_trace = fm.get_hyper_trace()
    last_alphs = hp_trace["alpha"].iloc[-20:].values
    assert np.all(last_alphs > ((1 / alpha_inv**2) / 2))
    assert np.all(last_alphs < ((1 / alpha_inv**2) * 2))

    last_w0_samples = assert_unwrap(fm.w0_samples)[-20:]
    assert np.all(last_w0_samples < (stub_weight.global_bias + 0.5))
    assert np.all(last_w0_samples > (stub_weight.global_bias - 0.5))

    last_w_samples = assert_unwrap(fm.w_samples)[-20:]

    for w_ in last_w_samples:
        assert np.all(w_ < (stub_weight.weight + 1.0))
        assert np.all(w_ > (stub_weight.weight - 1.0))

    last_V_samples = assert_unwrap(fm.V_samples)[-20:]

    for i in range(3):
        for j in range(i + 1, 3):
            cross_term = stub_weight.factors[:, i].dot(stub_weight.factors[:, j])
            if abs(cross_term) < 0.1:
                continue
            sign = cross_term / abs(cross_term)
            vfm_cross_term = vfm_weights.V[i].dot(vfm_weights.V[j])
            assert vfm_cross_term > sign * cross_term * 0.8
            assert vfm_cross_term < sign * cross_term * 1.25

            for V_ in last_V_samples:
                sample_cross_term = V_[i].dot(V_[j])
                assert sample_cross_term > sign * cross_term * 0.5
                assert sample_cross_term < sign * cross_term * 2
