import numpy as np
import pandas as pd
from pytest_mock import MockerFixture

from myfm.utils.benchmark_data import MovieLens100kDataManager


def test_ml100k(mocker: MockerFixture) -> None:
    mocker.patch("builtins.input", return_value="y")
    dm = MovieLens100kDataManager()
    unique_key_pair = ["user_id", "movie_id"]

    df_all_recovered = dm.load_rating_all().sort_values(unique_key_pair)

    user_infos = dm.load_user_info()
    assert np.all(df_all_recovered["user_id"].isin(user_infos["user_id"]))
    assert np.all(user_infos["age"] >= 0)
    assert np.all(user_infos["gender"].isin(["M", "F"]).values)

    movie_infos = dm.load_movie_info()
    genres = dm.genres()
    for genre_concat in movie_infos["genres"]:
        for genre in genre_concat.split("|"):
            assert genre in genres

    for k in [2, 3]:
        df_train, df_test = dm.load_rating_predefined_split(k)
        df_reconcat = pd.concat([df_train, df_test]).sort_values(unique_key_pair)
        for key in ["user_id", "movie_id", "timestamp"]:
            assert np.all(df_all_recovered[key].values == df_reconcat[key].values)

    N_manual_fold = 7
    df_tests = []
    for i in range(N_manual_fold):
        df_tr, df_te = dm.load_rating_kfold_split(N_manual_fold, i)
        assert (
            pd.concat([df_tr, df_te]).drop_duplicates(unique_key_pair).shape[0]
            == df_all_recovered.shape[0]
        )
        assert df_tr.shape[0] + df_te.shape[0] == df_all_recovered.shape[0]
        test_size = df_all_recovered.shape[0] // N_manual_fold
        assert df_te.shape[0] in {test_size, test_size + 1}
        df_tests.append(df_te)
    df_tests_concat = pd.concat(df_tests)
    assert df_tests_concat.shape[0] == df_all_recovered.shape[0]
    assert (
        df_tests_concat.drop_duplicates(unique_key_pair).shape[0]
        == df_all_recovered.shape[0]
    )
