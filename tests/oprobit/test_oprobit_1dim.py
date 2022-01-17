import numpy as np
from myfm import MyFMOrderedProbit


def test_oprobit() -> None:
    N_train = 200
    cps = np.asfarray([0.0, 0.5])
    rns = np.random.RandomState(0)
    X = rns.normal(0, 2, size=N_train)
    coeff = 0.5
    y = np.zeros(N_train, dtype=np.float64)
    score = X * coeff + rns.randn(N_train)
    for cp_value in cps:
        y += (score > cp_value).astype(np.int64)
    fm = MyFMOrderedProbit(0, fit_w0=False)
    fm.fit(X[:, None], y)
    for sample in fm.predictor_.samples[-10:]:
        cp_1, cp_2 = sample.cutpoints[0]
        assert abs(cp_1) < 0.25
        assert abs(cp_2 - cp_1 - 0.5) < 0.25

    prediction = fm.predict(X[:, None].astype(np.float32))
    assert np.all(prediction[np.where(X > 1.0)] == 2)