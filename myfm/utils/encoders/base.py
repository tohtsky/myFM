from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Tuple
from .many_to_many import ManyToManyEncoder

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


class DataFrameEncoder:
    """Encode pandas.DataFrame into concatenated sparse matrices."""

    def __init__(self) -> None:
        """Construct the encoders starting from empty one."""
        self.col_encoders: Dict[str, SparseEncoderBase] = OrderedDict()
        self.many_to_many_encoders: List[Tuple[str, str, str, ManyToManyEncoder]] = []

    @property
    def encoder_shapes(self) -> List[int]:
        """Show how the columns for an encoded CSR matrix are organized.

        Returns
        -------
        List[int]
            list of length of internal encoders.
        """
        return [len(enc) for enc in self.col_encoders.values()] + [
            len(mtomenc) for _, _, _, mtomenc in self.many_to_many_encoders
        ]

    def add_column(
        self, colname: str, encoder: SparseEncoderBase
    ) -> "DataFrameEncoder":
        """Add a column name to be encoded / encoder pair.

        Parameters
        ----------
        colname : str
            The column name to be encoded.
        encoder : SparseEncoderBase
            The corresponding encoder.
        """
        self.col_encoders[colname] = encoder
        return self

    def add_many_to_many(
        self,
        left_keyname: str,
        target_colname: str,
        encoder: ManyToManyEncoder,
        right_keyname: Optional[str] = None,
    ) -> "DataFrameEncoder":
        if right_keyname is None:
            right_keyname = left_keyname
        self.many_to_many_encoders.append(
            (left_keyname, right_keyname, target_colname, encoder)
        )
        return self

    def encode_df(
        self, df: pd.DataFrame, right_tables: List[pd.DataFrame] = []
    ) -> sps.csr_matrix:
        """Encode the dataframe into a concatenated CSR matrix.

        Parameters
        ----------
        df : pd.DataFrame
            The source.
        right_tables: List[pd.DataFrame]
            Tables to be taken "left-join" operation with df as the left table.
            You must feed as many tables as the many-to-many encoders you have already registered.

        Returns
        -------
        sps.csr_matrix
            The result.
        """
        matrices: List[sps.csr_matrix] = []
        for colname, encoder in self.col_encoders.items():
            matrices.append(encoder.to_sparse(df[colname]))
        for (
            left_key,
            right_key,
            target_colname,
            mtom_encoder,
        ), right_df in zip(self.many_to_many_encoders, right_tables):
            matrices.append(
                mtom_encoder.encode_df(
                    df, left_key, right_df, right_key, target_colname
                )
            )
        return sps.hstack(matrices, format="csr")
