import numpy as np
from myfm import MyFMOrderedProbit
from myfm.base import std_cdf
from myfm.utils.callbacks import OrderedProbitCallback


def test_oprobit() -> None:
    N_train = 500
    cps = np.asfarray([0.0, 0.5, 1.5])
    rns = np.random.RandomState(0)
    X = rns.normal(0, 2, size=N_train)
    coeff = 0.5
    y = np.zeros(N_train, dtype=np.float64)
    score = X * coeff + rns.randn(N_train)
    for cp_value in cps:
        y += (score > cp_value).astype(np.int64)
    callback = OrderedProbitCallback(100, X_test=X[:, None], y_test=y, n_class=4)
    fm = MyFMOrderedProbit(0, fit_w0=False)
    fm.fit(
        X[:, None],
        y,
        X_test=X,
        y_test=y,
        callback=callback,
        n_iter=100,
        n_kept_samples=100,
    )

    assert fm.predictor_ is not None
    for sample in fm.predictor_.samples[-10:]:
        cp_1, cp_2, cp_3 = sample.cutpoints[0]
        assert abs(cp_1) < 0.25
        assert abs(cp_2 - cp_1 - 0.5) < 0.25
        assert abs(cp_3 - cp_1 - 1.5) < 0.25

    prediction = fm.predict(X[:, None].astype(np.float32))

    p_using_core = fm.predict_proba(X[:, None])

    np.testing.assert_allclose(callback.predictions / 100, p_using_core)
    result_manual = np.zeros((X.shape[0], 4))

    n_ = 0
    for sample in fm.predictor_.samples:
        n_ += 1
        score = sample.predict_score(X[:, None], [])
        cdf = std_cdf((sample.cutpoints[0][np.newaxis, :] - score[:, np.newaxis]))
        diff = np.hstack(
            [
                np.zeros((score.shape[0], 1)),
                cdf,
                np.ones((score.shape[0], 1)),
            ]
        )
        result_manual += diff[:, 1:] - diff[:, :-1]
    result_manual /= n_
    np.testing.assert_allclose(result_manual, p_using_core)
