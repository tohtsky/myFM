from typing import Tuple

import numpy as np
import pytest
from scipy import sparse as sps

from myfm import MyFMGibbsClassifier, VariationalFMClassifier
from myfm.utils.callbacks import ClassificationCallback

from ..test_utils import FMWeights


@pytest.mark.parametrize("use_libfm_callback", [True, False])
def test_middle_clf(
    middle_data: Tuple[sps.csr_matrix, np.ndarray],
    stub_weight: FMWeights,
    use_libfm_callback: bool,
) -> None:
    rns = np.random.RandomState(0)
    X, score = middle_data
    score_noised = score + rns.normal(0, 1, size=score.shape)
    score_noised -= score_noised.mean()
    y = score_noised > 0
    if use_libfm_callback:
        callback = ClassificationCallback(200, X, y)
    else:
        callback = None

    fm = MyFMGibbsClassifier(3).fit(
        X, y, X_test=X, y_test=y, n_iter=200, n_kept_samples=200, callback=callback
    )
    if use_libfm_callback:
        np.testing.assert_allclose(fm.predict_proba(X), callback.predictions / 200)

    vfm_before_fit = VariationalFMClassifier(3)
    assert vfm_before_fit.w0 is None
    assert vfm_before_fit.w0_var is None
    assert vfm_before_fit.w is None
    assert vfm_before_fit.w_var is None
    assert vfm_before_fit.V is None
    assert vfm_before_fit.V_var is None

    vfm = vfm_before_fit.fit(
        X, y, X_test=X, y_test=y, n_iter=200  # , n_kept_samples=50
    )

    assert vfm.w0 is not None
    assert vfm.w0_var is not None
    assert vfm.w is not None
    assert vfm.w_var is not None
    assert vfm.V is not None
    assert vfm.V_var is not None

    assert fm.predictor_ is not None

    last_samples = fm.predictor_.samples[-20:]

    for i in range(3):
        for j in range(i + 1, 3):
            cross_term = stub_weight.factors[:, i].dot(stub_weight.factors[:, j])
            if abs(cross_term) < 0.5:
                continue
            sign = cross_term / abs(cross_term)
            assert vfm.V[i].dot(vfm.V[j]) > sign * cross_term * 0.8
            assert vfm.V[i].dot(vfm.V[j]) < sign * cross_term * 1.2

            for s in last_samples:
                sample_cross_term = s.V[i].dot(s.V[j])
                assert sample_cross_term > sign * cross_term * 0.5
                assert sample_cross_term < sign * cross_term * 2
