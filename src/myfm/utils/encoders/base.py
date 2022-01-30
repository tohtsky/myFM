from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List

import pandas as pd
import scipy.sparse as sps


class SparseEncoderBase(ABC):
    r"""The base class for encoders into sparse matrices."""

    @abstractmethod
    def to_sparse(self, x: List[Any]) -> sps.csr_matrix:
        raise NotImplementedError("must be implemented")  # pragma: no cover

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("must be implemented")  # pragma: no cover

    @abstractmethod
    def names(self) -> List[str]:
        r"""Description of each non-zero entry."""
        raise NotImplementedError("must be implemented")  # pragma: no cover


class DataFrameEncoder:
    """Encode pandas.DataFrame into concatenated sparse matrices."""

    def __init__(self) -> None:
        r"""Construct the encoders starting from empty one."""
        self.col_encoders: Dict[str, SparseEncoderBase] = OrderedDict()

    def all_names(self) -> List[str]:
        return [
            f"{col_name}__{description}"
            for col_name, encoder in self.col_encoders.items()
            for description in encoder.names()
        ]

    @property
    def encoder_shapes(self) -> List[int]:
        r"""Show how the columns for an encoded CSR matrix are organized.

        Returns
        -------
        List[int]
            list of length of internal encoders.
        """
        return [len(enc) for enc in self.col_encoders.values()]

    def add_column(
        self, colname: str, encoder: SparseEncoderBase
    ) -> "DataFrameEncoder":
        r"""Add a column name to be encoded / encoder pair.

        Parameters
        ----------
        colname : str
            The column name to be encoded.
        encoder : SparseEncoderBase
            The corresponding encoder.
        """
        self.col_encoders[colname] = encoder
        return self

    def encode_df(self, df: pd.DataFrame) -> sps.csr_matrix:
        r"""Encode the dataframe into a concatenated CSR matrix.

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
