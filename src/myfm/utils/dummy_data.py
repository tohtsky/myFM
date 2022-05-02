from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps

from myfm.base import DenseArray, RelationBlock


def gen_dummy_rating_df(
    random_seed: int = 0,
    factor_rank: int = 3,
    size: int = 100,
    user_colname: str = "userId",
    item_colname: str = "itemId",
    timestamp_colname: str = "timestamp",
    rating_colname: str = "rating",
) -> pd.DataFrame:
    rns = np.random.RandomState(random_seed)
    user_indices_all = np.arange(max(int(size / 3), 10))
    item_indices_all = np.arange(max(int(size / 2), 10))
    user_factor = rns.normal(
        0, 1 / factor_rank**0.5, size=(user_indices_all.shape[0], factor_rank)
    )
    item_factor = rns.normal(0, 1, size=(item_indices_all.shape[0], factor_rank))

    time = pd.Timestamp("2000-01-01") + pd.to_timedelta(
        rns.randint(-365, 365, size=size), unit="day"
    )

    result_df = pd.DataFrame(
        {
            user_colname: rns.choice(user_indices_all, size=size, replace=True) + 1,
            item_colname: rns.choice(item_indices_all, size=size, replace=True) + 1,
            timestamp_colname: time,
        }
    )
    score = (
        user_factor[result_df[user_colname].values - 1, :]
        * item_factor[result_df[item_colname].values - 1, :]
    ).sum(axis=1)
    cutpoints: List[float] = list(np.percentile(score, [20, 40, 60, 80]))  # type: ignore
    rating = np.ones((size,), dtype=np.int64)
    for cp in cutpoints:
        rating += score >= cp
    result_df[rating_colname] = rating
    return result_df


def gen_dummy_X(
    random_seed: int = 0,
    factor_rank: int = 3,
    size: int = 100,
) -> Tuple[List[RelationBlock], DenseArray, List[int]]:
    user_column = "userId"
    item_column = "itemId"
    rating_column = "rating"
    df_ = gen_dummy_rating_df(
        random_seed,
        factor_rank=factor_rank,
        size=size,
        user_colname=user_column,
        item_colname=item_column,
        rating_colname=rating_column,
    )
    blocks = []
    shapes = []
    for colname in [user_column, item_column]:
        categorical_expression = pd.Categorical(df_[colname])
        X = sps.identity(
            len(categorical_expression.categories), dtype=np.float64
        ).tocsr()
        ind = categorical_expression.codes
        blocks.append(RelationBlock(ind, X))
        shapes.append(X.shape[1])
    return (blocks, df_[rating_column].values, shapes)


__all__ = ["gen_dummy_rating_df"]
