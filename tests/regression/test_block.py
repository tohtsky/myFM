import numpy as np
from scipy import sparse as sps
from myfm import MyFMRegressor, RelationBlock, VariationalFMRegressor


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
    X_flatten_squread.data = X_flatten_squread.data ** 2

    weights = rns.randn(3, X_flatten.shape[1])
    Xw = X_flatten.dot(weights.T)
    X2w2 = X_flatten_squread.dot((weights ** 2).sum(axis=0))
    y = 0.5 * ((Xw ** 2).sum(axis=1) - X2w2) + rns.randn(N_train)

    blocks = [
        RelationBlock(user_indices, user_block),
        RelationBlock(item_indices, item_block),
    ]
    fm_flatten = MyFMRegressor(2, fit_w0=False, fit_linear=False).fit(
        X_flatten,
        y,
        group_shapes=group_shapes,
        n_iter=30,
        n_kept_samples=30,
    )
    fm_blocked = MyFMRegressor(2, fit_w0=False, fit_linear=False).fit(
        tm_column,
        y,
        blocks,
        group_shapes=group_shapes,
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
