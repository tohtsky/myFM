import pickle
import tempfile

import numpy as np
from scipy import sparse as sps

from myfm import MyFMRegressor, RelationBlock, VariationalFMRegressor


def test_block_vfm() -> None:
    N_train = 1000
    rns = np.random.RandomState(1)
    user_block = sps.csr_matrix(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
        dtype=np.float64,
    )
    user_indices = rns.randint(0, user_block.shape[0], size=N_train)
    item_block = sps.csr_matrix(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    item_indices = rns.randint(0, item_block.shape[0], size=N_train)
    tm_column = rns.randn(N_train, 1)

    X_flatten = sps.hstack(
        [tm_column, user_block[user_indices], item_block[item_indices]]
    )
    X_flatten_squread = X_flatten.copy()
    X_flatten_squread.data = X_flatten_squread.data**2
    factor = rns.randn(X_flatten.shape[1], 3)
    f2 = (factor**2).sum(axis=1)
    Xf = X_flatten.dot(factor)

    gb = 3.0
    linear_weights = rns.randn(X_flatten.shape[1])
    y = (
        gb
        + X_flatten.dot(linear_weights)
        + 0.5 * ((Xf**2).sum(axis=1) - X_flatten_squread.dot(f2))
        + rns.normal(1.0, size=X_flatten.shape[0])
    )

    blocks = [
        RelationBlock(user_indices, user_block),
        RelationBlock(item_indices, item_block),
    ]
    with tempfile.TemporaryFile() as temp_fs:
        pickle.dump(blocks, temp_fs)
        del blocks
        temp_fs.seek(0)
        blocks = pickle.load(temp_fs)
    fm_flatten = VariationalFMRegressor(3).fit(
        X_flatten,
        y,
        n_iter=100,
    )
    fm_blocked = VariationalFMRegressor(3).fit(
        tm_column,
        y,
        blocks,
        n_iter=100,
    )

    assert fm_flatten.predictor_ is not None
    assert fm_blocked.predictor_ is not None
    with tempfile.TemporaryFile() as temp_fs:
        pickle.dump(fm_blocked, temp_fs)
        del fm_blocked
        temp_fs.seek(0)
        fm_blocked = pickle.load(temp_fs)

    np.testing.assert_allclose(
        fm_flatten.predictor_.weights().w, fm_blocked.predictor_.weights().w
    )
    np.testing.assert_allclose(
        fm_flatten.predictor_.weights().V, fm_blocked.predictor_.weights().V
    )
    predicton_flatten = fm_flatten.predict(tm_column, blocks)
    predicton_blocked = fm_blocked.predict(X_flatten)
    np.testing.assert_allclose(predicton_flatten, predicton_blocked)


def test_block() -> None:
    rns = np.random.RandomState(0)
    N_train = 100
    user_block = sps.csr_matrix(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    user_indices = rns.randint(0, user_block.shape[0], size=N_train)
    item_block = sps.csr_matrix(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.float64,
    )

    group_shapes = [1, user_block.shape[1], item_block.shape[1]]
    item_indices = rns.randint(0, item_block.shape[0], size=N_train)
    tm_column = rns.randn(N_train, 1)

    X_flatten = sps.hstack(
        [tm_column, user_block[user_indices], item_block[item_indices]]
    )
    X_flatten_squread = X_flatten.copy()
    X_flatten_squread.data = X_flatten_squread.data**2

    weights = rns.randn(3, X_flatten.shape[1])
    Xw = X_flatten.dot(weights.T)
    X2w2 = X_flatten_squread.dot((weights**2).sum(axis=0))
    y = 0.5 * ((Xw**2).sum(axis=1) - X2w2) + rns.randn(N_train)

    blocks = [
        RelationBlock(user_indices, user_block),
        RelationBlock(item_indices, item_block),
    ]
    fm_flatten = MyFMRegressor(2, fit_w0=False).fit(
        X_flatten,
        y,
        group_shapes=group_shapes,
        n_iter=30,
        n_kept_samples=30,
    )
    fm_blocked = MyFMRegressor(2, fit_w0=False).fit(
        tm_column,
        y,
        blocks,
        group_shapes=group_shapes,
        n_iter=30,
        n_kept_samples=30,
    )
    assert fm_flatten.predictor_ is not None
    assert fm_blocked.predictor_ is not None
    for s_flatten, s_blocked in zip(
        fm_flatten.predictor_.samples, fm_blocked.predictor_.samples
    ):
        np.testing.assert_allclose(s_flatten.V, s_blocked.V)

    with tempfile.TemporaryFile() as temp_fs:
        pickle.dump(fm_blocked, temp_fs)
        del fm_blocked
        temp_fs.seek(0)
        fm_blocked = pickle.load(temp_fs)

    predicton_flatten = fm_flatten.predict(tm_column, blocks, n_workers=2)
    predicton_blocked = fm_blocked.predict(X_flatten, n_workers=None)
    np.testing.assert_allclose(predicton_flatten, predicton_blocked)
