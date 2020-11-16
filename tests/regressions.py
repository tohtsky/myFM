import pickle
import os
import unittest
from collections import defaultdict
from typing import Dict, List
from abc import ABC, abstractmethod
from unittest.case import TestCase

import numpy as np
import pandas as pd
from myfm import (
    MyFMGibbsClassifier,
    MyFMGibbsRegressor,
    RelationBlock,
    VariationalFMClassifier,
    VariationalFMRegressor,
    MyFMOrderedProbit,
)
from myfm.utils.benchmark_data import MovieLens100kDataManager
from myfm.utils.encoders import (
    BinningEncoder,
    CategoryValueToSparseEncoder,
    DataFrameEncoder,
    MultipleValuesToSparseEncoder,
)


class TestAll(TestCase):
    def setUp(self) -> None:
        data_manager = MovieLens100kDataManager()
        df_train, df_test = data_manager.load_rating_predefined_split(1)

        user_to_watched: Dict[int, List[int]] = defaultdict(list)
        movie_to_watched: Dict[int, List[int]] = defaultdict(list)

        datetime_encoder = BinningEncoder(
            df_train.timestamp.values.astype(np.int64), n_percentiles=200
        )

        user_unique = df_train[["user_id"]].drop_duplicates()
        movie_unique = df_train[["movie_id"]].drop_duplicates()

        for row in df_train.itertuples():
            uid = row.user_id
            mid = row.movie_id
            user_to_watched[uid].append(mid)
            movie_to_watched[mid].append(uid)

        self.user_to_watched = user_to_watched
        self.movie_to_watched = movie_to_watched

        self.user_unique = user_unique.set_index("user_id")
        self.movie_unique = movie_unique.set_index("movie_id")

        self.movie_encoders = DataFrameEncoder()
        self.movie_encoders.add_column(
            "movie_id", CategoryValueToSparseEncoder[int](movie_unique.movie_id)
        )
        self.movie_encoders.add_column(
            "movie_watched",
            MultipleValuesToSparseEncoder[int](
                list(movie_to_watched.values()), normalize_row=True
            ),
        )
        self.user_encoders = DataFrameEncoder()
        self.user_encoders.add_column(
            "user_id", CategoryValueToSparseEncoder[int](user_unique.user_id)
        )
        self.user_encoders.add_column(
            "user_watched",
            MultipleValuesToSparseEncoder[int](
                list(user_to_watched.values()), normalize_row=True
            ),
        )

        [
            (self.X_main_train, self.blocks_train, self.y_train),
            (self.X_main_test, self.blocks_test, self.y_test),
        ] = [
            (
                datetime_encoder.to_sparse(df.timestamp.values.astype(np.int64)),
                [
                    self.user_id_to_relation_block(df.user_id),
                    self.movie_id_to_relation_block(df.movie_id),
                ],
                df.rating.values - 1,  # for ordered probit
            )
            for df in [df_train, df_test]
        ]
        self.y_train_binary = self.y_train >= 4
        self.y_test_binary = self.y_test >= 4

    def user_id_to_relation_block(self, user_ids):
        uid_unique, index = np.unique(user_ids, return_inverse=True)
        df = self.user_unique.reindex(uid_unique).reset_index()
        df["user_watched"] = [self.user_to_watched[uid] for uid in df.user_id]
        X = self.user_encoders.encode_df(df)
        return RelationBlock(index, X)

    def movie_id_to_relation_block(self, movie_ids):
        mid_unique, index = np.unique(movie_ids, return_inverse=True)
        df = self.movie_unique.reindex(mid_unique).reset_index()
        df["movie_watched"] = [self.movie_to_watched[mid] for mid in df.movie_id]

        X = self.movie_encoders.encode_df(df)
        return RelationBlock(index, X)

    def test_main(self):
        for CLS, classification in [
            (MyFMGibbsRegressor, False),
            (MyFMGibbsClassifier, True),
            (VariationalFMRegressor, False),
            (VariationalFMClassifier, True),
            (MyFMOrderedProbit, False),
        ]:
            fm = CLS(rank=2, random_seed=43).fit(
                self.X_main_train,
                self.y_train if classification else self.y_train_binary,
                X_rel=self.blocks_train,
                X_test=self.X_main_test,
                y_test=self.y_test if classification else self.y_test_binary,
                X_rel_test=self.blocks_test,
                n_iter=10,
                n_kept_samples=10,
            )
            prediction_1 = fm.predict(self.X_main_test, self.blocks_test)

            with open("temp.pkl", "wb") as ofs:
                pickle.dump(fm, ofs)

            with open("temp.pkl", "rb") as ifs:
                fm_recovered = pickle.load(ifs)
            prediction_2 = fm_recovered.predict(self.X_main_test, self.blocks_test)

            self.assertTrue(np.all(prediction_1 == prediction_2))


if __name__ == "__main__":
    unittest.main()
