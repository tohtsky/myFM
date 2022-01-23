import argparse
import pickle
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

import myfm
from myfm import MyFMOrderedProbit, MyFMRegressor, RelationBlock
from myfm.gibbs import MyFMOrderedProbit
from myfm.utils.benchmark_data import MovieLens10MDataManager
from myfm.utils.callbacks.libfm import (
    LibFMLikeCallbackBase,
    OrderedProbitCallback,
    RegressionCallback,
)
from myfm.utils.encoders import CategoryValueToSparseEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    This script apply the method and evaluation protocal proposed in
    "On the Difficulty of Evaluating Baselines" paper by Rendle et al,
    against smaller Movielens 1M dataset, using myFM.
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "fold_index",
        type=int,
        help="which index set to use as a test within 10-fold CV.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["regression", "oprobit"],
        default="regression",
        help="specify the output type.",
    )
    parser.add_argument(
        "-i", "--iteration", type=int, help="mcmc iteration", default=512
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        help="fm embedding dimension",
        default=128,
    )
    parser.add_argument(
        "--stricter_protocol",
        action="store_true",
        help="Whether to use the \"stricter\" protocol (i.e., don't include the test set implicit information) stated in [Rendle, '19].",
        default=True,
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        choices=["mf", "svdpp", "timesvd", "timesvdpp", "timesvdpp_flipped"],
        help="feature set used in the experiment.",
        default="timesvdpp_flipped",
    )
    args = parser.parse_args()

    random_seed = 42

    # Additional features.
    # We add
    # 1. date of evaluation as categorical variables
    # 2. "all users who have evaluated a movie in the train set" or
    # 3. "all movies rated by a user" as a feature of user/movie.
    if args.feature == "mf":
        use_date = False
        use_iu = False
        use_ii = False
    elif args.feature == "svdpp":
        use_date = False
        use_iu = True
        use_ii = False
    elif args.feature == "timesvd":
        use_date = True
        use_iu = False
        use_ii = False
    elif args.feature == "timesvdpp":
        use_date = True
        use_iu = True
        use_ii = False
    elif args.feature == "timesvdpp_flipped":
        use_date = True  # use date info or not
        use_iu = True  # use implicit user feature
        use_ii = True  # use implicit item feature
    else:
        raise ValueError("unknown feature set specified.")

    FOLD_INDEX = args.fold_index
    ITERATION = args.iteration
    DIMENSION = args.dimension
    if FOLD_INDEX < 0 or FOLD_INDEX >= 10:
        raise ValueError("fold_index must be in the range(10).")
    ALGORITHM = args.algorithm
    data_manager = MovieLens10MDataManager()
    df_train, df_test = data_manager.load_rating_kfold_split(
        10, FOLD_INDEX, random_seed
    )

    if ALGORITHM == "oprobit":
        # interpret the rating 0.5, 1.0 ... , 5.0 as class (0, 1, ... , 10)
        for df_ in [df_train, df_test]:
            df_["rating"] -= 0.5
            df_["rating"] *= 2
            df_["rating"] = df_.rating.astype(np.int32)

    if args.stricter_protocol:
        implicit_data_source = df_train
    else:
        implicit_data_source = pd.concat([df_train, df_test])

    user_to_internal = CategoryValueToSparseEncoder[int](
        implicit_data_source.user_id.values
    )
    movie_to_internal = CategoryValueToSparseEncoder[int](
        implicit_data_source.movie_id.values
    )

    print(
        "df_train.shape = {}, df_test.shape = {}".format(df_train.shape, df_test.shape)
    )
    # treat the days of events as categorical variable
    date_encoder = CategoryValueToSparseEncoder[pd.Timestamp](
        implicit_data_source.timestamp.dt.date.values
    )

    def categorize_date(df):
        return date_encoder.to_sparse(df.timestamp.dt.date.values)

    movie_vs_watched: Dict[int, List[int]] = dict()
    user_vs_watched: Dict[int, List[int]] = dict()

    for row in implicit_data_source.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id
        movie_vs_watched.setdefault(movie_id, list()).append(user_id)
        user_vs_watched.setdefault(user_id, list()).append(movie_id)

    if use_date:
        X_date_train = categorize_date(df_train)
        X_date_test = categorize_date(df_test)
    else:
        X_date_train, X_date_test = (None, None)

    # setup grouping
    feature_group_sizes = []
    if use_date:
        feature_group_sizes.append(
            len(date_encoder),  # date
        )

    feature_group_sizes.append(len(user_to_internal))  # user ids

    if use_iu:
        # all movies which a user watched
        feature_group_sizes.append(len(movie_to_internal))

    feature_group_sizes.append(len(movie_to_internal))  # movie ids

    if use_ii:
        feature_group_sizes.append(
            len(user_to_internal)  # all the users who watched a movies
        )

    grouping = [i for i, size in enumerate(feature_group_sizes) for _ in range(size)]

    def augment_user_id(user_ids: List[int]) -> sps.csr_matrix:
        X = user_to_internal.to_sparse(user_ids)
        if not use_iu:
            return X
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []
        for index, user_id in enumerate(user_ids):
            watched_movies = user_vs_watched.get(user_id, [])
            normalizer = 1 / max(len(watched_movies), 1) ** 0.5
            for mid in watched_movies:
                data.append(normalizer)
                col.append(movie_to_internal[mid])
                row.append(index)
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(user_ids), len(movie_to_internal)),
                ),
            ],
            format="csr",
        )

    def augment_movie_id(movie_ids: List[int]):
        X = movie_to_internal.to_sparse(movie_ids)
        if not use_ii:
            return X

        data: List[float] = []
        row: List[int] = []
        col: List[int] = []

        for index, movie_id in enumerate(movie_ids):
            watched_users = movie_vs_watched.get(movie_id, [])
            normalizer = 1 / max(len(watched_users), 1) ** 0.5
            for uid in watched_users:
                data.append(normalizer)
                row.append(index)
                col.append(user_to_internal[uid])
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(movie_ids), len(user_to_internal)),
                ),
            ]
        )

    # Create RelationBlock.
    train_blocks: List[RelationBlock] = []
    test_blocks: List[RelationBlock] = []
    for source, target in [(df_train, train_blocks), (df_test, test_blocks)]:
        unique_users, user_map = np.unique(source.user_id, return_inverse=True)
        target.append(RelationBlock(user_map, augment_user_id(unique_users)))
        unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
        target.append(RelationBlock(movie_map, augment_movie_id(unique_movies)))

    trace_path = "rmse_{0}_fold_{1}.csv".format(ALGORITHM, FOLD_INDEX)

    callback: LibFMLikeCallbackBase
    fm: Union[MyFMRegressor, MyFMOrderedProbit]
    if ALGORITHM == "regression":
        fm = myfm.MyFMRegressor(rank=DIMENSION)
        callback = RegressionCallback(
            ITERATION,
            X_date_test,
            df_test.rating.values,
            X_rel_test=test_blocks,
            clip_min=0.5,
            clip_max=5.0,
            trace_path=trace_path,
        )
    else:
        fm = myfm.MyFMOrderedProbit(rank=DIMENSION)
        callback = OrderedProbitCallback(
            ITERATION,
            X_date_test,
            df_test.rating.values,
            n_class=10,
            X_rel_test=test_blocks,
            trace_path=trace_path,
        )
    fm.fit(
        X_date_train,
        df_train.rating.values,
        X_rel=train_blocks,
        grouping=grouping,
        n_iter=callback.n_iter,
        callback=callback,
        n_kept_samples=1,
    )
    with open(
        "callback_result_{0}_fold_{1}.pkl".format(ALGORITHM, FOLD_INDEX), "wb"
    ) as ofs:
        pickle.dump(callback, ofs)
