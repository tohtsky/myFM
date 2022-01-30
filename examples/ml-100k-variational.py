import argparse
from typing import List

import numpy as np
import pandas as pd

from myfm import RelationBlock, VariationalFMRegressor
from myfm.utils.benchmark_data.movielens100k_data import MovieLens100kDataManager
from myfm.utils.encoders import (
    CategoryValueToSparseEncoder,
    DataFrameEncoder,
    MultipleValuesToSparseEncoder,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    This script apply the method and evaluation protocal proposed in
    "On the Difficulty of Evaluating Baselines" paper by Rendle et al,
    against smaller Movielens 100K dataset, using myFM.
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "fold_index",
        type=int,
        help="which index set to use as a test within 5-fold predefined CV.",
        default=1,
    )

    parser.add_argument(
        "-i", "--iteration", type=int, help="mcmc iteration", default=512
    )
    parser.add_argument(
        "-d", "--dimension", type=int, help="fm embedding dimension", default=10
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
    if FOLD_INDEX < 1 or FOLD_INDEX >= 6:
        raise ValueError("fold_index must be in the range(1, 6).")

    data_manager = MovieLens100kDataManager()
    df_train, df_test = data_manager.load_rating_predefined_split(fold=FOLD_INDEX)

    if args.stricter_protocol:
        implicit_data_source = df_train
    else:
        implicit_data_source = pd.concat([df_train, df_test])

    def int_list_to_str(x):
        return "|".join([f"{id}" for id in x])

    user_implicit_profile = (
        implicit_data_source.groupby("user_id")["movie_id"]
        .agg(int_list_to_str)
        .reset_index()
    )
    item_implicit_profile = (
        implicit_data_source.groupby("movie_id")["user_id"]
        .agg(int_list_to_str)
        .reset_index()
    )

    print(
        "df_train.shape = {}, df_test.shape = {}".format(df_train.shape, df_test.shape)
    )

    user_encoder = DataFrameEncoder().add_column(
        "user_id",
        CategoryValueToSparseEncoder(user_implicit_profile.user_id),
    )
    if use_iu:
        user_encoder.add_column(
            "movie_id",
            MultipleValuesToSparseEncoder(user_implicit_profile.movie_id, sep="|"),
        )

    movie_encoder = DataFrameEncoder().add_column(
        "movie_id",
        CategoryValueToSparseEncoder(item_implicit_profile.movie_id),
    )
    if use_ii:
        movie_encoder.add_column(
            "user_id",
            MultipleValuesToSparseEncoder(item_implicit_profile.user_id, sep="|"),
        )

    # treat the days of events as categorical variable

    feature_group_sizes: List[int] = []
    if use_date:
        date_encoder = CategoryValueToSparseEncoder(
            implicit_data_source.timestamp.dt.date
        )
        X_date_train = date_encoder.to_sparse(df_train.timestamp.dt.date)
        X_date_test = date_encoder.to_sparse(df_test.timestamp.dt.date)
        feature_group_sizes.append(len(date_encoder))
    else:
        X_date_train, X_date_test = (None, None)

    # setup grouping
    feature_group_sizes.extend(user_encoder.encoder_shapes)
    feature_group_sizes.extend(movie_encoder.encoder_shapes)

    # Create RelationBlock.
    train_blocks: List[RelationBlock] = []
    test_blocks: List[RelationBlock] = []
    for source, target in [(df_train, train_blocks), (df_test, test_blocks)]:
        unique_users, user_map = np.unique(source.user_id, return_inverse=True)
        target.append(
            RelationBlock(
                user_map,
                user_encoder.encode_df(
                    user_implicit_profile.set_index("user_id")
                    .reindex(unique_users)
                    .fillna("")
                    .reset_index()
                ),
            )
        )
        unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
        target.append(
            RelationBlock(
                movie_map,
                movie_encoder.encode_df(
                    item_implicit_profile.set_index("movie_id")
                    .reindex(unique_movies)
                    .fillna("")
                    .reset_index()
                ),
            )
        )

    trace_path = "rmse_variational_fold_{0}.csv".format(FOLD_INDEX)
    fm = VariationalFMRegressor(rank=DIMENSION)

    fm.fit(
        X_date_train,
        df_train.rating.values,
        X_rel=train_blocks,
        n_iter=ITERATION,
        group_shapes=feature_group_sizes,
    )
    rmse = (
        (df_test.rating.values - fm.predict(X_date_test, test_blocks)) ** 2
    ).mean() ** 0.5
    assert fm.history_ is not None
    print("RMSE = {rmse}".format(rmse=rmse))
