import numpy as np
import pandas as pd

from myfm.utils.encoders import (
    BinningEncoder,
    DataFrameEncoder,
    MultipleValuesToSparseEncoder,
)
from myfm.utils.encoders.categorical import CategoryValueToSparseEncoder


def test_dfe() -> None:
    rns = np.random.RandomState(0)
    N = 1000
    categories = [1, 2, 3]

    multi_categories = ["i1", "i2", "i3", "i4"]
    multi_values = []
    cnts = []
    for _ in range(N):
        n = rns.randint(0, len(multi_categories) + 1)
        cnts.append(n)
        v = ",".join(rns.choice(multi_categories, size=n, replace=False))
        multi_values.append(v)
    df = pd.DataFrame(
        dict(
            numerical_value=rns.randn(N),
            categorical_value=rns.choice(categories, size=N, replace=True),
            multi_values=multi_values,
        )
    )
    dfe = DataFrameEncoder().add_column(
        "numerical_value", BinningEncoder(df.numerical_value)
    )
    assert np.all(dfe.encode_df(df).sum(axis=1).A1 == 1.0)
    dfe.add_column(
        "categorical_value", CategoryValueToSparseEncoder(df.categorical_value)
    )
    assert np.all(dfe.encode_df(df).sum(axis=1).A1 == 2.0)
    dfe.add_column(
        "multi_values", MultipleValuesToSparseEncoder(df.multi_values, normalize=False)
    )
    for nnz, cnt in zip(dfe.encode_df(df).sum(axis=1).A1, cnts):
        assert nnz == cnt + 2
    cursor = 0
    names = dfe.all_names()
    for s, name_prefix in zip(
        dfe.encoder_shapes, ["numerical_value", "categorical_value", "multi_values"]
    ):
        for X_col_name in names[cursor : cursor + s]:
            assert X_col_name.startswith(name_prefix)
        cursor += s
