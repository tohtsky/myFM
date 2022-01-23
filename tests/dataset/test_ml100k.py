import numpy as np
import pandas as pd
from pytest_mock import MockerFixture

from myfm.utils.benchmark_data import MovieLens100kDataManager


def test_ml100k(mocker: MockerFixture) -> None:
    mocker.patch("builtins.input", return_value="y")
    dm = MovieLens100kDataManager()

    df_all_recovered = dm.load_rating_all().sort_values(["user_id", "movie_id"])
    user_infos = dm.load_user_info()
    assert np.all(df_all_recovered["user_id"].isin(user_infos["user_id"]))
    assert np.all(user_infos["age"] >= 0)
    assert np.all(user_infos["gender"].isin(["M", "F"]).values)
    for k in [2, 3]:
        df_train, df_test = dm.load_rating_predefined_split(k)
        df_reconcat = pd.concat([df_train, df_test]).sort_values(
            ["user_id", "movie_id"]
        )
        for key in ["user_id", "movie_id", "timestamp"]:
            assert np.all(df_all_recovered[key].values == df_reconcat[key].values)
