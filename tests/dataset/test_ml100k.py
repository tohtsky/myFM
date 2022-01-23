from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pytest
from pytest_mock import MockerFixture
from sklearn.model_selection import KFold

from myfm.utils.benchmark_data import MovieLens100kDataManager
from myfm.utils.dummy_data import gen_dummy_rating_df


def test_ml100k(mocker: MockerFixture) -> None:
    df = gen_dummy_rating_df(user_colname="user_id", item_colname="movie_id")
    df["timestamp"] = (df["timestamp"].values.astype(int) / 1e9).astype(int)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "ml-100k.zip"
        mocker.patch("builtins.input", return_value="N")
        with pytest.raises(RuntimeError):
            MovieLens100kDataManager()
        mocker.patch("builtins.input", return_value="y")

        def patch_callable() -> None:
            with ZipFile(temp_path, mode="w") as zf:
                zf.writestr(
                    "ml-100k/u.data",
                    df[["user_id", "movie_id", "rating", "timestamp"]].to_csv(
                        sep="\t", header=None
                    ),
                )

        mocker.patch("urllib.request.urlretrieve", new_callable=patch_callable)
        dm = MovieLens100kDataManager(temp_path)

    df_all_recovered = dm.load_rating_all()
    for key in ["user_id", "movie_id"]:
        assert np.all(df_all_recovered[key].values == df[key].values)
