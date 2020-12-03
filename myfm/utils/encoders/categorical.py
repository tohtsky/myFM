from collections import Counter
from typing import Generic, List, TypeVar, Union

import numpy as np
import scipy.sparse as sps

from .base import SparseEncoderBase

T = TypeVar("T")


class CategoryValueToSparseEncoder(Generic[T], SparseEncoderBase):
    """The class to one-hot encode a List of items into a sparse matrix representation."""

    def __init__(self, items: List[T], min_freq: int = 1):
        """Construct the encoder by providing the known item set.
        It has a position for "unknown or too rare" items,
        which are regarded as the 0-th class.

        Parameters
        ----------
        items : List[T]
            The items list.
        min_freq : int, optional
            The minimal frequency for an item to be retained in the known items list, by default 1
        """
        counter_ = Counter(items)
        unique_items = [x for x, freq in counter_.items() if freq >= min_freq]
        self._dict = {item: i + 1 for i, item in enumerate(unique_items)}
        self._items: List[Union[str, T]] = ["UNK"]
        self._items.extend(unique_items)

    def __getitem__(self, x: T) -> int:
        return self._dict.get(x, 0)

    def names(self) -> List[Union[str, T]]:
        return self._items

    def to_sparse(self, items: List[T]) -> sps.csr_matrix:
        cols = [self[j] for j in items]
        return sps.csr_matrix(
            (
                np.ones(len(items), dtype=np.float64),
                (
                    np.arange(
                        len(items),
                    ),
                    cols,
                ),
            ),
            shape=(len(items), len(self)),
        )

    def __len__(self) -> int:
        return len(self._dict) + 1
