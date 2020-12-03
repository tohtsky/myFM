from collections import Counter
from typing import Generic, List, TypeVar

import numpy as np
import pandas as pd
import scipy.sparse as sps

T = TypeVar("T")


class ManyToManyEncoder(Generic[T]):
    def __init__(self, items: List[T], min_freq: int = 1, normalize: bool = True):
        counter = Counter(items)
        self.unique_items = [item for item, cnt in counter.items() if cnt >= min_freq]
        self.item_to_index = {item: i + 1 for i, item in enumerate(self.unique_items)}
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.item_to_index) + 1

    def encode_df(
        self,
        left_table: pd.DataFrame,
        left_key: str,
        right_table: pd.DataFrame,
        right_key: str,
        target_colname: str,
    ) -> sps.csr_matrix:
        unique_keys, inverse = np.unique(left_table[left_key], return_inverse=True)
        unique_key_to_index = {key: i for i, key in enumerate(unique_keys)}
        right_table_restricted = right_table[right_table[right_key].isin(unique_keys)]
        row = right_table_restricted[right_key].map(unique_key_to_index).values
        col = (
            right_table_restricted[target_colname]
            .map(self.item_to_index)
            .fillna(0.0)
            .astype(np.int64)
            .values
        )
        X = sps.csr_matrix(
            (
                np.ones(len(right_table_restricted), dtype=np.float64),
                (row, col),
            ),
            shape=(len(unique_keys), len(self)),
        )
        X.sort_indices()

        if self.normalize:
            norm = X.power(2).sum(axis=1).A1 ** 0.5
            X.data = X.data / norm[X.nonzero()[0]]
        X = X[inverse]
        X.sort_indices()
        return X
