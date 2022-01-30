import numpy as np

from myfm.utils.encoders import BinningEncoder


def test_binning_dense() -> None:
    rns = np.random.RandomState(0)
    v = rns.randn(1000)
    v[0] = np.nan
    enc = BinningEncoder(v)
    assert enc.percentiles.shape[0] == 10
    X = enc.to_sparse(v)
    assert np.all(X.sum(axis=1).A1 == 1.0)
    assert X.shape[1] == 12
    assert X[0, 0] == 1.0
    for j in np.where(v[1:] <= enc.percentiles[0])[0]:
        assert X[j + 1, 1] == 1.0
    for k in range(1, 10):
        for j in np.where(
            (v[1:] <= enc.percentiles[k]) & (v[1:] > enc.percentiles[k - 1])
        )[0]:
            assert X[j + 1, k + 1] == 1.0
    for j in np.where(v[1:] > enc.percentiles[-1])[0]:
        assert X[j + 1, 11] == 1.0


def test_binning_sparse() -> None:
    rns = np.random.RandomState(0)
    v = rns.poisson(2, size=1000)
    enc = BinningEncoder(v)
    X = enc.to_sparse(v)
    assert np.all(X.sum(axis=1).A1 == 1.0)
    assert X.shape[1] == len(enc)
    for j in np.where(v == 0)[0]:
        assert X[j, 1] == 1.0

    for j in np.where(v == 1.0)[0]:
        assert X[j, 2] == 1.0

    for k in range(2, len(enc.percentiles)):
        for j in np.where((v <= enc.percentiles[k]) & (v > enc.percentiles[k - 1]))[0]:
            assert X[j, k + 1] == 1.0

    for j in np.where(v > enc.percentiles[-1])[0]:
        assert X[j, len(enc.percentiles) + 1] == 1.0
