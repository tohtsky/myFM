from typing import Tuple
from scipy import sparse as sps
import numpy as np
from ..util import FMWeights

from myfm import MyFMGibbsClassifier


def test_middle_clf(
    middle_data: Tuple[sps.csr_matrix, np.ndarray],
    stub_weight: FMWeights,
) -> None:
    rns = np.random.RandomState(0)
    X, score = middle_data
    score_noised = score + rns.normal(0, 1, size=score.shape)
    score_noised -= score_noised.mean()
    y = score_noised > 0

    fm = MyFMGibbsClassifier(3).fit(
        X, score_noised, X_test=X, y_test=y, n_iter=200  # , n_kept_samples=50
    )

    last_samples = fm.predictor_.samples[-20:]

    for i in range(3):
        for j in range(i + 1, 3):
            cross_term = stub_weight.factors[:, i].dot(stub_weight.factors[:, j])
            for s in last_samples:
                sample_cross_term = s.V[i].dot(s.V[j])
                if cross_term > 0:
                    assert sample_cross_term > cross_term * 0.5
                    assert sample_cross_term < cross_term * 2
                elif cross_term < 0:
                    assert sample_cross_term < cross_term * 0.5
                    assert sample_cross_term > cross_term * 2
