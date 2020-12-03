import os
import pickle
import unittest
from unittest.case import TestCase

import numpy as np
from myfm import (
    MyFMGibbsClassifier,
    MyFMGibbsRegressor,
    MyFMOrderedProbit,
    RelationBlock,
    VariationalFMClassifier,
    VariationalFMRegressor,
)
from myfm.utils.benchmark_data import MovieLens100kDataManager
from myfm.utils.encoders import (
    BinningEncoder,
    CategoryValueToSparseEncoder,
    DataFrameEncoder,
    ManyToManyEncoder,
)
from sklearn import metrics

ITERATION = 10
RANK = 4


class TestAll(TestCase):
    def setUp(self) -> None:
        data_manager = MovieLens100kDataManager()
        df_train, df_test = data_manager.load_rating_predefined_split(1)

        self.df_train = df_train

        datetime_encoder = BinningEncoder(
            df_train.timestamp.values.astype(np.int64), n_percentiles=200
        )

        user_unique = df_train[["user_id"]].drop_duplicates()
        movie_unique = df_train[["movie_id"]].drop_duplicates()

        self.user_unique = user_unique.set_index("user_id")
        self.movie_unique = movie_unique.set_index("movie_id")

        self.movie_encoders = (
            DataFrameEncoder()
            .add_column("movie_id", CategoryValueToSparseEncoder(movie_unique.movie_id))
            .add_many_to_many(
                "movie_id",
                "user_id",
                ManyToManyEncoder(user_unique.user_id, normalize=True),
            )
        )

        self.user_encoders = (
            DataFrameEncoder()
            .add_column("user_id", CategoryValueToSparseEncoder(user_unique.user_id))
            .add_many_to_many(
                "user_id",
                "movie_id",
                ManyToManyEncoder(movie_unique.movie_id, normalize=True),
            )
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
        X = self.user_encoders.encode_df(
            self.user_unique.reindex(uid_unique).reset_index(),
            right_tables=[self.df_train],
        )
        return RelationBlock(index, X)

    def movie_id_to_relation_block(self, movie_ids):
        mid_unique, index = np.unique(movie_ids, return_inverse=True)
        X = self.movie_encoders.encode_df(
            self.movie_unique.reindex(mid_unique).reset_index(), [self.df_train]
        )
        return RelationBlock(index, X)

    def test_main(self):
        for CLS, problem in [
            (MyFMGibbsRegressor, "reg"),
            (MyFMGibbsClassifier, "clf"),
            (VariationalFMRegressor, "reg"),
            (VariationalFMClassifier, "clf"),
            (MyFMOrderedProbit, "ord"),
        ]:
            fm = CLS(rank=RANK, random_seed=42).fit(
                self.X_main_train,
                self.y_train_binary if problem == "clf" else self.y_train,
                X_rel=self.blocks_train,
                X_test=self.X_main_test,
                y_test=self.y_test_binary if problem == "clf" else self.y_test,
                X_rel_test=self.blocks_test,
                n_iter=ITERATION,
                n_kept_samples=ITERATION,
            )

            # test pickling
            if problem == "reg":
                prediction_1 = fm.predict(self.X_main_test, self.blocks_test)
            else:
                prediction_1 = fm.predict_proba(self.X_main_test, self.blocks_test)

            with open("temp.pkl", "wb") as ofs:
                pickle.dump(fm, ofs)

            with open("temp.pkl", "rb") as ifs:
                fm_recovered = pickle.load(ifs)

            os.remove("temp.pkl")
            if problem == "reg":
                prediction_2 = fm_recovered.predict(self.X_main_test, self.blocks_test)
            else:
                prediction_2 = fm_recovered.predict_proba(
                    self.X_main_test, self.blocks_test
                )

            self.assertTrue(np.all(prediction_1 == prediction_2))

            if problem == "reg":
                rmse = ((self.y_test - prediction_1) ** 2).mean() ** 0.5
                print("rmse={}".format(rmse))
            elif problem == "clf":
                roc = metrics.roc_auc_score(self.y_test_binary, prediction_1)
                print("roc={}".format(roc))
            elif problem == "ord":
                ll = metrics.log_loss(self.y_test, prediction_1)
                print("log loss={}".format(ll))


if __name__ == "__main__":
    unittest.main()
