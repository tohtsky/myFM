from numpy.lib.function_base import percentile
from .base import SparseEncoderBase
from scipy import sparse as sps
from typing import List
from numbers import Number
import numpy as np


class BinningEncoder(SparseEncoderBase):
    """The class to one-hot encode a List of numerical values into a sparse matrix representation by binning."""

    def __init__(self, x: List[Number], n_percentiles: int = 10) -> None:
        """Initializes the encoder by compting the percentile values of input.

        Parameters
        ----------
        x : List[Number]
            list of numerical values.
        n_percentiles : int, optional
            number of percentiles computed against x, by default 10.
        """
        self.percentages = np.linspace(0, 100, n_percentiles)
        self.percentiles = np.unique(np.percentile(x, self.percentages))

    def to_sparse(self, x: List[Number]) -> sps.csr_matrix:
        x_array = np.asfarray(x)
        N = x_array.shape[0]
        non_na_index = ~np.isnan(x_array)
        x_not_na = x_array[non_na_index]
        cols = np.zeros(N, dtype=np.int64)
        cols[non_na_index] += 1
        for p in self.percentiles:
            cols[non_na_index] += x_not_na >= p
        return sps.csr_matrix(
            (np.ones(N, dtype=np.float64), (np.arange(N), cols)),
            shape=(N, len(self)),
        )

    def __len__(self) -> int:
        return len(self.percentiles) + 2
