from io import BytesIO
from pathlib import Path

import pandas as pd

from .movielens100k_data import MovieLensBase


def read_ml1m10m_df(ifs: BytesIO) -> pd.DataFrame:
    r"""A hacky function to read from Movielens 1M/10M dataset using native parser.
    This hack is taken from irspack: https://github.com/tohtsky/irspack/blob/a1893be54200b0dc765957220deeccc1764fe39c/irspack/dataset/movielens/ML1M.py
    """
    df = pd.read_csv(
        ifs,
        sep=":",
        header=None,
    )[[0, 2, 4, 6]].copy()

    df.columns = ["user_id", "movie_id", "rating", "timestamp"]
    df["timestamp"] = pd.to_datetime(df.timestamp, unit="s")
    return df


class MovieLens1MDataManager(MovieLensBase):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DEFAULT_PATH = Path("~/.ml-1m.zip").expanduser()

    def load_rating_all(self) -> pd.DataFrame:
        """Read all (1M) interactions.

        Returns
        -------
        pd.DataFrame
            Movielens 1M rating dataframe.
        """
        with BytesIO(self.zf.read("ml-1m/ratings.dat")) as ifs:
            return read_ml1m10m_df(ifs)
