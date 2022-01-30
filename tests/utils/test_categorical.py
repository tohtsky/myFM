import numpy as np
import pytest

from myfm.utils.encoders import CategoryValueToSparseEncoder

TEST_ITEMS = [
    "item1",
    "item2",
    "item3",
    "item1",
    "item2",
    "item3",
    "item1",
    "item2",
]


def test_categorical_encs_create() -> None:
    enc = CategoryValueToSparseEncoder(TEST_ITEMS, handle_unknown="create")
    X = enc.to_sparse(["item4", "item1", "item2", "item3"])
    for i in range(4):
        for j in range(len(enc)):
            if i == j:
                assert X[i, j] == 1
            else:
                assert X[i, j] == 0

    enc_cutoff = CategoryValueToSparseEncoder(
        TEST_ITEMS, handle_unknown="create", min_freq=3
    )
    assert len(enc_cutoff) == 3
    X_cutoffed = enc_cutoff.to_sparse(["item4", "item1", "item2", "item3"])
    np.testing.assert_allclose(
        X_cutoffed.toarray(), np.asfarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    )


def test_categorical_encs_ignore() -> None:
    enc = CategoryValueToSparseEncoder(TEST_ITEMS, handle_unknown="ignore")
    X = enc.to_sparse(["item4", "item1", "item2", "item3"])
    np.testing.assert_allclose(
        X.toarray(), np.asfarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    enc_cutoff = CategoryValueToSparseEncoder(
        TEST_ITEMS, handle_unknown="ignore", min_freq=3
    )
    X = enc_cutoff.to_sparse(["item4", "item1", "item2", "item3"])
    np.testing.assert_allclose(
        X.toarray(), np.asfarray([[0, 0], [1, 0], [0, 1], [0, 0]])
    )


def test_categorical_encs_raise() -> None:
    enc = CategoryValueToSparseEncoder(TEST_ITEMS, handle_unknown="raise")
    with pytest.raises(KeyError):
        X = enc.to_sparse(["item4", "item1", "item2", "item3"])
    X = enc.to_sparse(["item1", "item2", "item3"])

    np.testing.assert_allclose(
        X.toarray(), np.asfarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
