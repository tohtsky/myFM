import numpy as np
from collections import Counter
from typing import List, Generic, TypeVar, Dict, Union
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
        self._dict: Dict[T, int] = dict()
        counter_ = Counter(items)
        unique_items = [x for x, freq in counter_.items() if freq >= min_freq]
        self._dict.update({item: i + 1 for i, item in enumerate(unique_items)})
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


class MultipleValuesToSparseEncoder(Generic[T], SparseEncoderBase):
    """The class to n-hot encode a List of List of items into a sparse matrix representation."""

    def __init__(
        self,
        items: List[List[T]],
        min_freq: int = 1,
        normalize_row: bool = False,
    ):
        """Construct the encoder by providing the known item set.
        It has a position for "unknown or too rare" items,
        which are regarded as the 0-th class.

        Parameters
        ----------
        items : List[List[T]]
            The list of list of items to construct encoders.
        min_freq : int, optional
            The minimal frequency for an item to be retained in the known items list, by default 1
        normalize_row: bool, optional
            Whether to normalize the output so that the l2-norm of each row becomes 1, by default False.
        """
        self._dict: Dict[T, int] = dict()
        self.normalize_row = normalize_row
        counter_ = Counter([y for x in items for y in x])
        unique_items = [x for x, freq in counter_.items() if freq >= min_freq]
        self._dict.update({item: i + 1 for i, item in enumerate(unique_items)})
        self._items: List[Union[str, T]] = ["UNK"]
        self._items.extend(unique_items)

    def __getitem__(self, x: T) -> int:
        return self._dict.get(x, 0)

    def names(self) -> List[Union[str, T]]:
        return self._items

    def to_sparse(self, items: List[List[T]]) -> sps.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        for i, row in enumerate(items):
            for item in row:
                rows.append(i)
                cols.append(self[item])
        result = sps.csr_matrix(
            (
                np.ones(len(rows), dtype=np.float64),
                (
                    rows,
                    cols,
                ),
            ),
            shape=(len(items), len(self)),
        )
        result.sort_indices()
        if not self.normalize_row:
            return result

        norms = result.power(2).sum(axis=1).A1 ** 0.5
        result.data /= norms[result.nonzero()[0]]
        return result

    def __len__(self) -> int:
        return len(self._dict) + 1
