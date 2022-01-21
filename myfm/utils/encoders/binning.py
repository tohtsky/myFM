from .base import SparseEncoderBase
from myfm.base import DenseArray
from scipy import sparse as sps
from typing import List, TypeVar, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = object

Numeric = TypeVar("Numeric", int, float)


class BinningEncoder(SparseEncoderBase):
    """The class to one-hot encode a List of numerical values into a sparse matrix representation by binning."""

    def __init__(self, x: ArrayLike, n_percentiles: int = 10) -> None:
        """Initializes the encoder by compting the percentile values of input.

        Parameters
        ----------
        x:
            list of numerical values.
        n_percentiles:
            number of percentiles computed against x, by default 10.

        """
        if n_percentiles <= 0:
            raise ValueError("n_percentiles must be greater than 0.")
        self.percentages = np.linspace(0, 100, n_percentiles)
        temp_percentiles: DenseArray = np.percentile(x, self.percentages)  # type: ignore
        self.percentiles = np.unique(temp_percentiles)  # type: ignore

    def names(self) -> List[str]:
        return (
            ["NaN"]
            + [f"<={val}" for val in self.percentiles]
            + [f">{self.percentiles[-1]}"]
        )

    def to_sparse(self, x: ArrayLike) -> sps.csr_matrix:
        x_array = np.asarray(x, dtype=np.float64)
        N = x_array.shape[0]
        non_na_index = ~np.isnan(x_array)
        x_not_na = x_array[non_na_index]
        cols = np.zeros(N, dtype=np.int64)
        cols[non_na_index] += 1
        for p in self.percentiles:
            cols[non_na_index] += x_not_na > p
        return sps.csr_matrix(
            (np.ones(N, dtype=np.float64), (np.arange(N), cols)),
            shape=(N, len(self)),
        )

    def __len__(self) -> int:
        return len(self.percentiles) + 2
