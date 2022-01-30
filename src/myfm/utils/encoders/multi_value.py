from typing import Iterable

import scipy.sparse as sps
from typing_extensions import Literal

from .categorical import CategoryValueToSparseEncoder


class MultipleValuesToSparseEncoder(CategoryValueToSparseEncoder[str]):
    """The class to N-hot encode a List of items into a sparse matrix representation."""

    def __init__(
        self,
        items: Iterable[str],
        min_freq: int = 1,
        sep: str = ",",
        normalize: bool = True,
        handle_unknown: Literal["create", "ignore", "raise"] = "create",
    ):
        """Construct the encoder by providing a list of strings,
        each of which is a list of strings concatenated by `sep`.

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
        handle_unknown: Literal["create", "ignore", "raise"], optional
            How to handle previously unseen values during encoding.
            If "create", then there is a single category named "__UNK__" for unknown values,
            ant it is treated as 0th category.
            If "ignore", such an item will be ignored.
            If "raise", a `KeyError` is raised.
            Defaults to "create".
        """
        items_flatten = [
            y for x in items for y in set(x.split(sep)) if y
        ]  # ignore empty string.
        self.sep = sep
        self.normalize = normalize
        super().__init__(
            items_flatten, min_freq=min_freq, handle_unknown=handle_unknown
        )

    def to_sparse(self, items: Iterable[str]) -> sps.csr_matrix:
        indptr = [0]
        indices = []
        data = []
        n_row = 0
        cursor = 0
        for row in items:
            n_row += 1
            items = row.split(self.sep)
            indices_local = sorted(
                list(
                    {
                        index
                        for index in [self._get_index(v) for v in items if v]
                        if index is not None
                    }
                )
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
