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

    vfm = VariationalFMClassifier(3).fit(
        X, y, X_test=X, y_test=y, n_iter=200  # , n_kept_samples=50
    )
    assert fm.predictor_ is not None

    last_samples = fm.predictor_.samples[-20:]

    for i in range(3):
        for j in range(i + 1, 3):
            cross_term = stub_weight.factors[:, i].dot(stub_weight.factors[:, j])
            m = vfm.predictor_.weights()
            if abs(cross_term) < 0.5:
                continue
            sign = cross_term / abs(cross_term)
            assert m.V[i].dot(m.V[j]) > sign * cross_term * 0.8
            assert m.V[i].dot(m.V[j]) < sign * cross_term * 1.2

            for s in last_samples:
                sample_cross_term = s.V[i].dot(s.V[j])
                assert sample_cross_term > sign * cross_term * 0.5
                assert sample_cross_term < sign * cross_term * 2
