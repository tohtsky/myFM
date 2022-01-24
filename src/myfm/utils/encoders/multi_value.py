from collections import Counter
from typing import Iterable, List

import scipy.sparse as sps

from .base import SparseEncoderBase


class MultipleValueToSparseEncoder(SparseEncoderBase):
    """The class to N-hot encode a List of items into a sparse matrix representation."""

    def __init__(
        self,
        items: Iterable[str],
        min_freq: int = 1,
        sep: str = ",",
        normalize: bool = True,
    ):
        """Construct the encoder by providing the known item set.
        It has a position for "unknown or too rare" items,
        which are regarded as the 0-th class.

        Parameters
        ----------
        items : Iterable[str]
            Iterable of strings, each of which is a concatenated list of possibly multiple items.
        min_freq : int, optional
            The minimal frequency for an item to be retained in the known items list, by default 1.
        sep: str, optional
            Tells how to separate string back into a list. Defaults to `','`.
        normalize: bool, optional
            If `True`, non-zero entry in the encoded matrix will have `1 / N ** 0.5`,
            where `N` is the number of non-zero entries in that row. Defaults to `True`.
        """
        items_flatten = [
            y for x in items for y in set(x.split(sep)) if y
        ]  # ignore empty string.
        counter_ = Counter(items_flatten)
        unique_items = [x for x, freq in counter_.items() if freq >= min_freq]
        self._dict = {item: i + 1 for i, item in enumerate(unique_items)}
        self._items: List[str] = ["UNK"]
        self._items.extend(unique_items)
        self.sep = sep
        self.normalize = normalize

    def __getitem__(self, x: str) -> int:
        return self._dict.get(x, 0)

    def names(self) -> List[str]:
        return self._items

    def to_sparse(self, items: Iterable[str]) -> sps.csr_matrix:
        indptr = [0]
        indices = []
        data = []
        n_row = 0
        cursor = 0
        for row in items:
            n_row += 1
            indices_local = sorted(
                list(set([self[v] for v in row.split(self.sep) if v]))
            )
            if not indices_local:
                indptr.append(cursor)
                continue
            n = len(indices_local)
            value = 1.0 / (float(n) ** 0.5) if self.normalize else 1.0
            indices.extend(indices_local)
            data.extend([value] * n)
            cursor += n
            indptr.append(cursor)
        return sps.csr_matrix(
            (data, indices, indptr),
            shape=(n_row, len(self)),
        )

    def __len__(self) -> int:
        return len(self._dict) + 1
