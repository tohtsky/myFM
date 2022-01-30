from collections import Counter
from typing import Dict, Generic, Iterable, List, Optional, TypeVar, Union

import numpy as np
import scipy.sparse as sps
from typing_extensions import Literal

from .base import SparseEncoderBase

T = TypeVar("T", int, float, str)


class CategoryValueToSparseEncoder(Generic[T], SparseEncoderBase):
    """The class to one-hot encode a List of items into a sparse matrix representation."""

    def __init__(
        self,
        items: Iterable[T],
        min_freq: int = 1,
        handle_unknown: Literal["create", "ignore", "raise"] = "create",
    ):
        r"""Construct the encoder by providing a list of items.

        Parameters
        ----------
        items : Iterable[T]
            The items list.
        min_freq : int, optional
            The minimal frequency for an item to be retained in the known items list, by default 1
        handle_unknown: Literal["create", "ignore", "raise"], optional
            How to handle previously unseen values during encoding.
            If "create", then there is a single category named "__UNK__" for unknown values,
            ant it is treated as 0th category.
            If "ignore", such an item will be ignored.
            If "raise", a `KeyError` is raised.
            Defaults to "create".
        """
        counter_ = Counter(items)
        unique_items = sorted([x for x, freq in counter_.items() if freq >= min_freq])
        self._item_index_offset = 1 if handle_unknown == "create" else 0
        self.handle_unknown = handle_unknown
        self._dict: Dict[T, int] = {
            item: i + self._item_index_offset for i, item in enumerate(unique_items)
        }
        self.values: List[Union[str, T]] = []
        if self.handle_unknown == "create":
            self.values.append("__UNK__")
        self.values.extend(unique_items)

    def _get_index(self, x: T) -> Optional[int]:
        try:
            return self._dict[x]
        except KeyError:
            if self.handle_unknown == "create":
                return 0
            elif self.handle_unknown == "ignore":
                return None
            raise

    def names(self) -> List[str]:
        return [str(y) for y in self.values]

    def to_sparse(self, items: Iterable[T]) -> sps.csr_matrix:
        rows = []
        cols = []
        n_row = 0
        for i, x in enumerate(items):
            n_row += 1
            index = self._get_index(x)
            if index is None:
                continue
            rows.append(i)
            cols.append(index)
        return sps.csr_matrix(
            (
                np.ones(len(rows), dtype=np.float64),
                (rows, cols),
            ),
            shape=(n_row, len(self)),
        )

    def __len__(self) -> int:
        return len(self._dict) + self._item_index_offset
