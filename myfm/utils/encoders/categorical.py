import numpy as np
from typing import List, Generic, TypeVar, Dict, Union
import scipy.sparse as sps
from .base import SparseEncoderBase

T = TypeVar("T")


class CategoryValueToSparseEncoder(Generic[T], SparseEncoderBase):
    def __init__(self, items: List[T]):
        self._dict: Dict[T, int] = dict()
        unique_items = list(set(items))
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
                np.ones(len(items)),
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
