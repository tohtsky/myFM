import numpy as np

from myfm.utils.encoders import MultipleValuesToSparseEncoder

TEST_ITEMS = [
    "item1",
    "item2, item1",
    "item3, item2",
    "item2",
    "item3, item1",
]


def test_categorical_encs_create() -> None:
    enc = MultipleValuesToSparseEncoder(TEST_ITEMS, handle_unknown="create")
    X = enc.to_sparse(["item4,item1", "item1,item2,item3", "item2", "item3"])
    nnz_rows = (X.toarray() > 0).astype(np.int32).sum(axis=1)
    np.testing.assert_allclose(nnz_rows, np.asarray([2, 3, 1, 1]))
