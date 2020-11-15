from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, List, Dict

import pandas as pd
import scipy.sparse as sps


class SparseEncoderBase(ABC):
    """The base class for sparse encoder"""

    @abstractmethod
    def to_sparse(self, x: List[Any]) -> sps.csr_matrix:
        raise NotImplementedError("must be implemented")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("must be implemented")


class DataFrameEncoder(object):
    """Encode pandas.DataFrame into concatenated sparse matrices."""

    def __init__(self):
        """Construct the encoders starting from empty one."""
        self.col_encoders: Dict[str, SparseEncoderBase] = OrderedDict()

    @property
    def encoder_shapes(self) -> List[int]:
        """Show how the columns for an encoded CSR matrix are organized.

        Returns
        -------
        List[int]
            list of length of internal encoders.
        """
        return [len(enc) for enc in self.col_encoders.items()]

    def add_column(self, colname: str, encoder: SparseEncoderBase) -> None:
        """Add a column name to be encoded / encoder pair.

        Parameters
        ----------
        colname : str
            The column name to be encoded.
        encoder : SparseEncoderBase
            The corresponding encoder.
        """
        self.col_encoders[colname] = encoder

    def encode_df(self, df: pd.DataFrame) -> sps.csr_matrix:
        """Encode the dataframe into a concatenated CSR matrix.

        Parameters
        ----------
        df : pd.DataFrame
            The source.

        Returns
        -------
        sps.csr_matrix
            The result.
        """
        matrices: List[sps.csr_matrix] = []
        for colname, encoder in self.col_encoders.items():
            matrices.append(encoder.to_sparse(df[colname]))
        return sps.hstack(matrices, format="csr")
