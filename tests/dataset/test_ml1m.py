import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pytest
from pytest_mock import MockerFixture

from myfm.utils.benchmark_data import MovieLens1MDataManager
from myfm.utils.dummy_data import gen_dummy_rating_df


def test_ml1m(mocker: MockerFixture) -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")
    dummy_df = gen_dummy_rating_df(user_colname="user_id", item_colname="movie_id")
    dummy_df["timestamp"] = (dummy_df["timestamp"].view(np.int64) / 1e9).astype(
        np.int64
    )
    with TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "ml1m.zip"
        mocker.patch("builtins.input", return_value="NOO")
        with pytest.raises(RuntimeError):
            dm = MovieLens1MDataManager(target)
        df_stringified = "\n".join(
            [
                "::".join([str(v) for v in row])
                for row in dummy_df[
                    ["user_id", "movie_id", "rating", "timestamp"]
                ].values
            ]
        )
        with ZipFile(target, "w") as zf:
            zf.writestr("ml-1m/ratings.dat", df_stringified)
        dm = MovieLens1MDataManager(target)
        unique_key_pair = ["user_id", "movie_id", "rating"]
        df_all_recovered = dm.load_rating_all()
        for key in unique_key_pair:
            assert np.all(df_all_recovered[key] == dummy_df[key])
