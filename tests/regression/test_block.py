import numpy as np
from numpy import random
from scipy import sparse as sps
from myfm import MyFMRegressor, RelationBlock


def test_block() -> None:
    rns = np.random.RandomState(0)
    N_train = 30
    user_block = sps.csr_matrix(
        [
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.float64,
    )
    user_indices = rns.randint(0, user_block.shape[0], size=N_train)
    item_block = sps.csr_matrix(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    item_indices = rns.randint(0, item_block.shape[0], size=N_train)
    tm_column = rns.randn(N_train, 1)

    X_flatten = sps.hstack(
        [tm_column, user_block[user_indices], item_block[item_indices]]
    )

    blocks = [
        RelationBlock(user_indices, user_block),
        RelationBlock(item_indices, item_block),
    ]
    y = rns.randn(N_train)
    fm_flatten = MyFMRegressor(2).fit(
        X_flatten, y, group_shapes=[1, 4, 3], n_iter=30, n_kept_samples=30
    )

    fm_blocked = MyFMRegressor(2).fit(
        tm_column,
        y,
        blocks,
        group_shapes=[1, 4, 3],
        n_iter=30,
        n_kept_samples=30,
    )
    for s_flatten, s_blocked in zip(
        fm_flatten.predictor_.samples, fm_blocked.predictor_.samples
    ):
        np.testing.assert_allclose(s_flatten.V, s_blocked.V)
    predicton_flatten = fm_flatten.predict(tm_column, blocks, n_workers=2)
    predicton_blocked = fm_blocked.predict(X_flatten, n_workers=None)
    np.testing.assert_allclose(predicton_flatten, predicton_blocked)
