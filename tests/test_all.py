import os
import pickle
import pytest

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
from typing import Tuple
import pandas as pd

ITERATION = 10
RANK = 4


def test_all() -> None:
    data_manager = MovieLens100kDataManager()
    df_train, df_test = data_manager.load_rating_predefined_split(1)

    datetime_encoder = BinningEncoder(
        df_train.timestamp.values.astype(np.int64), n_percentiles=200
    )

    user_unique = df_train[["user_id"]].drop_duplicates()
    movie_unique = df_train[["movie_id"]].drop_duplicates()

    user_unique = user_unique.set_index("user_id")
    movie_unique = movie_unique.set_index("movie_id")

    movie_encoders = (
        DataFrameEncoder()
        .add_column("movie_id", CategoryValueToSparseEncoder(movie_unique.movie_id))
        .add_many_to_many(
            "movie_id",
            "user_id",
            ManyToManyEncoder(user_unique.user_id, normalize=True),
        )
    )

    user_encoders = (
        DataFrameEncoder()
        .add_column("user_id", CategoryValueToSparseEncoder(user_unique.user_id))
        .add_many_to_many(
            "user_id",
            "movie_id",
            ManyToManyEncoder(movie_unique.movie_id, normalize=True),
        )
    )

    def user_id_to_relation_block(user_ids: np.ndarray) -> RelationBlock:
        uid_unique, index = np.unique(user_ids, return_inverse=True)
        X = user_encoders.encode_df(
            user_unique.reindex(uid_unique).reset_index(),
            right_tables=[df_train],
        )
        return RelationBlock(index, X)

    def movie_id_to_relation_block(movie_ids: np.ndarray) -> RelationBlock:
        mid_unique, index = np.unique(movie_ids, return_inverse=True)
        X = movie_encoders.encode_df(
            movie_unique.reindex(mid_unique).reset_index(), [df_train]
        )
        return RelationBlock(index, X)

    [(X_main_train, blocks_train, y_train), (X_main_test, blocks_test, y_test),] = [
        (
            datetime_encoder.to_sparse(df.timestamp.values.astype(np.int64)),
            [
                user_id_to_relation_block(df.user_id),
                movie_id_to_relation_block(df.movie_id),
            ],
            df.rating.values - 1,  # for ordered probit
        )
        for df in [df_train, df_test]
    ]
    y_train_binary = y_train >= 4
    y_test_binary = y_test >= 4

    for CLS, problem in [
        (MyFMGibbsRegressor, "reg"),
        (MyFMGibbsClassifier, "clf"),
        (VariationalFMRegressor, "reg"),
        (VariationalFMClassifier, "clf"),
        (MyFMOrderedProbit, "ord"),
    ]:
        fm = CLS(rank=RANK, random_seed=42).fit(
            X_main_train,
            y_train_binary if problem == "clf" else y_train,
            X_rel=blocks_train,
            X_test=X_main_test,
            y_test=y_test_binary if problem == "clf" else y_test,
            X_rel_test=blocks_test,
            n_iter=ITERATION,
            n_kept_samples=ITERATION,
        )

        # test pickling
        if problem == "reg":
            prediction_1 = fm.predict(X_main_test, blocks_test)
        else:
            prediction_1 = fm.predict_proba(X_main_test, blocks_test)

        with open("temp.pkl", "wb") as ofs:
            pickle.dump(fm, ofs)

        with open("temp.pkl", "rb") as ifs:
            fm_recovered = pickle.load(ifs)

        os.remove("temp.pkl")
        if problem == "reg":
            prediction_2 = fm_recovered.predict(X_main_test, blocks_test)
        else:
            prediction_2 = fm_recovered.predict_proba(X_main_test, blocks_test)

        assert np.all(prediction_1 == prediction_2)

        if problem == "reg":
            rmse = ((y_test - prediction_1) ** 2).mean() ** 0.5
            print("rmse={}".format(rmse))
        elif problem == "clf":
            roc = metrics.roc_auc_score(y_test_binary, prediction_1)
            print("roc={}".format(roc))
        elif problem == "ord":
            ll = metrics.log_loss(y_test, prediction_1)
            print("log loss={}".format(ll))
